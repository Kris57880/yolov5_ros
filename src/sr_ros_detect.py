#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from rostopic import get_topic_type

from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes

ROOT2 = Path(__file__).resolve().parents[2] / 'sr_ros/src'
if str(ROOT2) not in sys.path:
    sys.path.insert(0, str(ROOT2))

# import for liif and classifier
import torch.nn as nn
from PIL import Image as PilImage
import torchvision
from torchvision import transforms
from models2 import make
from utils2 import make_coord
from test import batched_predict

#for path in sys.path:
#	print(path)

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0] / "yolov5"
ROOT = FILE.parents[0] / "tph-yolov5"
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path
    
# import from yolov5 submodules
# from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from models.experimental import attempt_load

NAMES =['Fishing_Boat_1', 'Fishing_Boat_2', 'Fishing_Boat_3', 'Fishing_Boat_4', 'Hovercraft', 'Military_Boat_1', 'Military_Boat_2', 'Military_Boat_3', 'Yacht_1', 'Yacht_2']

@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.view_image = rospy.get_param("~view_image")
        # Initialize weights 
        weights = rospy.get_param("~yolo_weights")
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        # self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        if weights.endswith('.pt'):
            w = str(weights[0] if isinstance(weights, list) else weights)
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=self.device, fuse=True)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
                            
            self.stride, self.names, self.pt, self.jit, self.onnx, self.engine  = (
                stride,
                self.model.names,
                True,
                False,
                False,
                False,
            )
            print(stride, self.model.names)
        elif weights.endswith('.engine'):
            from tensorrt_helper import TRTModel
            model = TRTModel(weights,self.device)
            self.model = model
            self.stride, self.names, self.pt, self.jit, self.onnx, self.engine  = (
                32,
                NAMES, 
                False,
                False,
                False,
                True,
            )
        elif weights.endswith('.onnx'):
            check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            self.session = onnxruntime.InferenceSession(weights, None)
            self.stride, self.names, self.pt, self.jit, self.onnx, self.engine  = (
                32,
                NAMES, 
                False,
                False,
                True,
                False,
            )
    
        


        # ==========Initialize LIIF and classifier==========
        self.liif = make(
            torch.load(rospy.get_param('~liif_weights'))['model'], load_sd=True).cuda()
        self.classifier = torchvision.models.resnet18(pretrained=False)
        self.classifier.fc = nn.Linear(512, 10)
        self.classifier.load_state_dict(torch.load(rospy.get_param('~classifier_weights')))
        self.classifier.to(self.device)
        self.enable_liif = rospy.get_param("~enable_liif")
        # for param in self.liif.parameters():
        #     print(param.data)
        # ==================================================

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        #self.img_size = [rospy.get_param("~inference_size_h",480), rospy.get_param("~inference_size_w", 640)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # self.model.warmup()  # warmup       
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.callback, queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.callback, queue_size=1
            )

        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        )
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=10
            )
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()


    def callback(self, data):
        """adapted from yolov5/detect.py"""
        # print(data.header)
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        else:
            im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        im, im0 = self.preprocess(im)
        im2 = im0.copy()
        # print(im.shape)
        # print(img0.shape)
        # print(img.shape)

        # Run inference
        if self.pt or self.jit or self.engine:
            im = torch.from_numpy(im).to(self.device) 
            im = im.half() if self.half else im.float()
        if self.onnx:
            im = im.astype('float16') if self.half else  im.astype('float32') 
        
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        if self.pt or self.jit:
            pred = self.model(im, augment=False, visualize=False)[0]
        elif self.engine:
            pred = self.model(im)[-1]
            # pred = pred.reshape(1,1,-1,pred.shape[-1])
        elif self.onnx:
            pred = torch.tensor(self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im}))

        # print(pred)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                # Fill in bounding box message
                bounding_box.Class = self.names[c]
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])

                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       

                
                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()   

        # Publish prediction
        if self.enable_liif:
            new_boxes, im2 = self.liif_classify(bounding_boxes, im, im2)
            self.pred_pub.publish(new_boxes)
        else:
            self.pred_pub.publish(bounding_boxes)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow("from YOLO", im0)
            #cv2.resizeWindow("from YOLO", 1422, 800) 
            if self.enable_liif:
                cv2.imshow("from LIIF & classifier", im2)
                #cv2.resizeWindow("from LIIF & classifier", 1422, 800)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            if self.enable_liif:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(im2, "bgr8"))
            else:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0 
        

    def liif_classify(self, bounding_boxes, img, img2):
        # print(img.shape)    # BCHW
        # _, _, img_h, img_w = img.shape
        # full_h, full_w, _ = img2.shape	# 1080, 1920, 3
        # hscale = float(img_h) / float(full_h)
        # wscale = float(img_w) / float(full_w)
        
        img3 = img2.copy()
        # Convert
        img3 = img3[..., ::-1]  # BGR to RGB
        
        img3 = PilImage.fromarray(img3)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img3 = transform(img3).to(self.device)
        
        
        new_boxes = BoundingBoxes()
        new_boxes.header = bounding_boxes.header
        new_boxes.image_header = bounding_boxes.image_header
        annotator = Annotator(img2, line_width=self.line_thickness, example=str(self.names))
        rospy.loginfo(f"# of bounding_boxes:{len(bounding_boxes.bounding_boxes)}")

        self.classifier.eval()
        with torch.no_grad():
            for idx, bounding_box in enumerate(bounding_boxes.bounding_boxes):
                #if idx > 0: break
                rospy.loginfo(f'Bounding box {idx}: {bounding_box.Class}')
                #rospy.loginfo(f"full img shape: {img3.shape}")
                #rospy.loginfo(f"actual img shape: {img.shape}")
                rospy.loginfo(f'({bounding_box.xmin}, {bounding_box.ymin}), ({bounding_box.xmax}, {bounding_box.ymax})')
                #rospy.loginfo(f'real:({bounding_box.xmin* wscale:.2f}, {bounding_box.ymin* hscale:.2f}), ({bounding_box.xmax* wscale:.2f}, {bounding_box.ymax* hscale:.2f})')
                
                cropped_img = torchvision.transforms.functional.crop(
                    img3, bounding_box.ymin, bounding_box.xmin, 
                    bounding_box.ymax - bounding_box.ymin, bounding_box.xmax - bounding_box.xmin)
                    
                #rospy.loginfo(f"crop shape: {cropped_img.shape}")
                #cropped_bgr = cv2.cvtColor(cropped_img.permute(1, 2, 0).cpu().numpy()*255, cv2.COLOR_RGB2BGR)
                #cv2.imshow(f"crop box-{str(idx)}", cropped_bgr.astype(np.uint8))
                rospy.loginfo(f"box max min: {cropped_img.max()} {cropped_img.min()}")
                
                h = w = 300
                coord = make_coord((h, w)).cuda()
                cell = torch.ones_like(coord)
                cell[:, 0] *= 2 / h
                cell[:, 1] *= 2 / w
                pred = batched_predict(self.liif, ((cropped_img - 0.5) / 0.5).unsqueeze(0),
                    coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
                pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1)	# BCHW to HWC to CHW 
                #pred_bgr = cv2.cvtColor(pred.permute(1, 2, 0).cpu().numpy()*255, cv2.COLOR_RGB2BGR)
                #cv2.imshow(f"pred-{str(idx)}", pred_bgr.astype(np.uint8))
                rospy.loginfo(f"pred max min: {pred.max()} {pred.min()}")	# 0~1
                rospy.loginfo(f"shape: {pred.shape}")	# 3 300 300            

                outputs = self.classifier(pred.unsqueeze(0))
                conf, prediction = torch.max(outputs, 1)

                bounding_box.Class = self.names[prediction.item()]
                new_boxes.bounding_boxes.append(bounding_box)
                rospy.loginfo(f'After classification: {self.names[prediction.item()]}, {conf.item():.2f}')
                rospy.loginfo(f'{outputs}')

                if self.publish_image or self.view_image:  # Add bbox to image
                    # integer class
                    #label = f"{bounding_box.Class} {conf.item():.2f}"
                    label = f"{bounding_box.Class}"
                    xyxy = (bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax)
                    annotator.box_label(xyxy, label, color=colors(prediction.item(), True))

        return new_boxes, annotator.result()


if __name__ == "__main__":
    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    
    rospy.spin()
