#!/usr/bin/env python3

# import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
import time
# from rostopic import get_topic_type

# from sensor_msgs.msg import Image, CompressedImage
# from detection_msgs.msg import BoundingBox, BoundingBoxes

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

class BoundingBox:
    def __init__(self, cls=None, probability=None, xmin=None, ymin=None, xmax=None, ymax=None):
        self.Class = cls
        self.probability = probability
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

class BoundingBoxes:
    def __init__(self):
        self.bounding_boxes = []

# @torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = 0.2
        self.iou_thres = 0.45
        self.agnostic_nms = True
        self.max_det = 1000
        self.classes = None
        self.line_thickness = 3
        self.view_image = False
        # Initialize weights 
        weights = f"{ROOT}/0311multi_classes_boat.engine"
        # Initialize model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        if weights.endswith('.pt'):
            w = str(weights[0] if isinstance(weights, list) else weights)
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=self.device, fuse=True)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
                            
            # self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            #     self.model.stride,
            #     self.model.names,
            #     self.model.pt,
            #     self.model.jit,
            #     self.model.onnx,
            #     self.model.engine,
            # )

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
                ['Fishing_Boat_1', 'Fishing_Boat_2', 'Fishing_Boat_3', 'Fishing_Boat_4', 'Hovercraft', 'Military_Boat_1', 'Military_Boat_2', 'Military_Boat_3', 'Yacht_1', 'Yacht_2'],
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
                ['Fishing_Boat_1', 'Fishing_Boat_2', 'Fishing_Boat_3', 'Fishing_Boat_4', 'Hovercraft', 'Military_Boat_1', 'Military_Boat_2', 'Military_Boat_3', 'Yacht_1', 'Yacht_2'],
                False,
                False,
                True,
                False,
            )
        # ==========Initialize LIIF and classifier==========
        self.liif = make(
            torch.load(f"{ROOT2}/model_weight/1019_epoch-best.pth")['model'], load_sd=True).cuda()
        self.classifier = torchvision.models.resnet18(pretrained=False)
        self.classifier.fc = nn.Linear(512, 10)
        self.classifier.load_state_dict(torch.load(f"{ROOT2}/model_weight/classifier_epoch-100.pth"))
        self.classifier.to(self.device)
        self.enable_liif = True

        self.target = 'Hovercraft'
        self.counter = 0
        self.box_filter = 1000

        for param in self.liif.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        # ==================================================

        # Setting inference size
        self.img_size = [2016, 2016]
        #self.img_size = [rospy.get_param("~inference_size_h",480), rospy.get_param("~inference_size_w", 640)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = False
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # self.model.warmup()  # warmup       
        
        # Initialize subscriber to Image/CompressedImage topic
        # input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        # self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        # if self.compressed_input:
        #     self.image_sub = rospy.Subscriber(
        #         input_image_topic, CompressedImage, self.callback, queue_size=1
        #     )
        # else:
        #     self.image_sub = rospy.Subscriber(
        #         input_image_topic, Image, self.callback, queue_size=1
        #     )

        # # Initialize prediction publisher
        # self.pred_pub = rospy.Publisher(
        #     rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        # )
        # # Initialize image publisher
        # self.publish_image = rospy.get_param("~publish_image")
        # if self.publish_image:
        #     self.image_pub = rospy.Publisher(
        #         rospy.get_param("~output_image_topic"), Image, queue_size=10
        #     )
        
        # # Initialize CV_Bridge
        # self.bridge = CvBridge()

    @torch.no_grad()
    def callback(self, data):
        """adapted from yolov5/detect.py"""
        # print(data.header)
        # if self.compressed_input:
        #     im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        # else:
        #     im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        im, im0 = self.preprocess(data)
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

        # with torch.no_grad():
        print('---> yolo time start')
        # print(next(self.model.parameters()).device)
        start_time = time.time()
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
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"---> yolo execution time:{execution_time}s per image")
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        # bounding_boxes.header = data.header
        # bounding_boxes.image_header = data.header
        
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
                #box_size = (bounding_box.ymax-bounding_box.ymin) * (bounding_box.xmax-bounding_box.xmin)

                # Annotate the image
                # if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    #if box_size > 100:
                    #	label = ""
                    #else:
                    #	label = f"{self.names[c]} {conf:.2f} {box_size}"
                label = f"{self.names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))       

                
                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()   

        # Publish prediction
        if self.enable_liif:
            bounding_boxes, im2 = self.liif_classify(bounding_boxes, im, im2)
            # self.pred_pub.publish(bounding_boxes)
        # else:
            # self.pred_pub.publish(bounding_boxes)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow("from YOLO", im0)
            cv2.resizeWindow("from YOLO", 1422, 800) 
            if self.enable_liif:
                cv2.imshow("from LIIF & classifier", im2)
                cv2.resizeWindow("from LIIF & classifier", 1422, 800)
            cv2.waitKey(1)  # 1 millisecond
            
        # if self.publish_image:
        #     if self.enable_liif:
        #         self.image_pub.publish(self.bridge.cv2_to_imgmsg(im2, "bgr8"))
        #     else:
        #         self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))

        # Auto terminate
        found = False
        for box in bounding_boxes.bounding_boxes:
            if self.enable_liif == False:
                box_size = (box.ymax-box.ymin) * (box.xmax-box.xmin)
                print(f"{box.Class} box_size: {box_size}")
            if str(box.Class) == str(self.target) and found == False:
                self.counter += 1
                print(f'Found {str(box.Class)}=={str(self.target)} {self.counter} times continuously.')
                found = True
                # break
        if found == False: self.counter = 0
        print(f'count:{self.counter}, found:{found}')
        if self.counter >= 5:
            print(f'Detected {str(self.target)} at box[({box.xmin}, {box.ymin}), ({box.xmax}, {box.ymax})], exiting.')
            #rospy.signal_shutdown(f'Found target.')
            #sys.exit(f"Detected {str(self.target)} at box[({box.xmin}, {box.ymin}), ({box.xmax}, {box.ymax})], exiting.")
        

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
        
        names = ['Fishing_Boat_1', 'Fishing_Boat_2', 'Fishing_Boat_3', 'Fishing_Boat_4', 'Hovercraft', 
                 'Military_Boat_1','Military_Boat_2','Military_Boat_3','Yacht_1','Yacht_2']  # class names
        
        new_boxes = BoundingBoxes()
        # new_boxes.header = bounding_boxes.header
        # new_boxes.image_header = bounding_boxes.image_header
        annotator = Annotator(img2, line_width=self.line_thickness, example=str(names))
        print(f"# of bounding_boxes:{len(bounding_boxes.bounding_boxes)}")

        self.classifier.eval()
        # print(next(self.liif.parameters()).device)
        # print(next(self.classifier.parameters()).device)
        with torch.no_grad():
            for idx, bounding_box in enumerate(bounding_boxes.bounding_boxes):
                #if idx > 0: break
                ##print(f'Bounding box {idx}: {bounding_box.Class}')
                #print(f"full img shape: {img3.shape}")
                #print(f"actual img shape: {img.shape}")
                ##print(f'({bounding_box.xmin}, {bounding_box.ymin}), ({bounding_box.xmax}, {bounding_box.ymax})')
                #print(f'real:({bounding_box.xmin* wscale:.2f}, {bounding_box.ymin* hscale:.2f}), ({bounding_box.xmax* wscale:.2f}, {bounding_box.ymax* hscale:.2f})')
                
                cropped_img = torchvision.transforms.functional.crop(
                    img3, bounding_box.ymin, bounding_box.xmin, 
                    bounding_box.ymax - bounding_box.ymin, bounding_box.xmax - bounding_box.xmin)
                box_size = (bounding_box.ymax-bounding_box.ymin) * (bounding_box.xmax-bounding_box.xmin)

                #print(f"crop shape: {cropped_img.shape}")
                #cropped_bgr = cv2.cvtColor(cropped_img.permute(1, 2, 0).cpu().numpy()*255, cv2.COLOR_RGB2BGR)
                #cv2.imshow(f"crop box-{str(idx)}", cropped_bgr.astype(np.uint8))
                ##print(f"box max min: {cropped_img.max()} {cropped_img.min()}")
                if box_size > self.box_filter:
                    print(f"skip box_size: {box_size}")
                    continue
                else:
                    h = w = 300
                    coord = make_coord((h, w)).cuda()
                    cell = torch.ones_like(coord)
                    cell[:, 0] *= 2 / h
                    cell[:, 1] *= 2 / w

                    print('---> liif time start')
                    start_time = time.time()

                    pred = batched_predict(self.liif, ((cropped_img - 0.5) / 0.5).unsqueeze(0),
                        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
                    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1)	# BCHW to HWC to CHW 
                    #pred_bgr = cv2.cvtColor(pred.permute(1, 2, 0).cpu().numpy()*255, cv2.COLOR_RGB2BGR)
                    #cv2.imshow(f"pred-{str(idx)}", pred_bgr.astype(np.uint8))
                    ##print(f"pred max min: {pred.max()} {pred.min()}")	# 0~1
                    ##print(f"shape: {pred.shape}")	# 3 300 300            

                    end_time = time.time()
                    print(f"---> liif execution time：{end_time-start_time}s")
                    
                    print('---> classifier time start')
                    start_time = time.time()

                    outputs = self.classifier(pred.unsqueeze(0))
                    conf, prediction = torch.max(outputs, 1)

                    end_time = time.time()
                    print(f"---> classifier execution time：{end_time-start_time}s")
                    
                    bounding_box.Class = names[prediction.item()]
                    new_boxes.bounding_boxes.append(bounding_box)
                    print(f"{bounding_box.Class} box_size: {box_size}")
                    ##print(f'After classification: {names[prediction.item()]}, {conf.item():.2f}')
                    ##print(f'{outputs}')

                    if self.view_image:  # Add bbox to image
                        # integer class
                        #label = f"{bounding_box.Class} {conf.item():.2f}"
                        label = f"{bounding_box.Class}"
                        xyxy = (bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax)
                        annotator.box_label(xyxy, label, color=colors(prediction.item(), True))

        return new_boxes, annotator.result()


if __name__ == "__main__":
    # check_requirements(exclude=("tensorboard", "thop"))
    
    # rospy.init_node("yolov5", anonymous=True)
    # detector = Yolov5Detector()
    
    # rospy.spin()


    # print(torch.backends.cudnn.enabled)
    # print(torch.backends.cudnn.benchmark)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    detector = Yolov5Detector()

    image_path = f"{ROOT2}/Fly_to_Target_01500.png"
    # image_path = f'{ROOT2}/No1_01890.png'
    # image_path = f"{ROOT2}/Loiter_3km_Zoom_1_10000.png"
    img = cv2.imread(image_path)

    exc_list = []
    if img is not None:
        for _ in range(10):
            start_time = time.time()
            detector.callback(img)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Total execution time：{execution_time}s per image")
            exc_list.append(execution_time)

        #cv2.imshow("Detected Image", img)
        print(f'AVG: {sum(exc_list)/len(exc_list):.4f}')
        print(f'AVG without 1st: {sum(exc_list[1:])/len(exc_list[1:]):.4f}')
        print(f'FPS without 1st: {1/(sum(exc_list[1:])/len(exc_list[1:])):.4f}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to load image.")
