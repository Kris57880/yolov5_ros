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


# add yolov5 submodule to path
FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0] / "yolov5"
ROOT = FILE.parents[0] / "tph-yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
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

# object tracking
import skimage
from sort import *

#.................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
	color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
	return tuple(color)


"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
	bbox_left = min([xyxy[0].item(), xyxy[2].item()])
	bbox_top = min([xyxy[1].item(), xyxy[3].item()])
	bbox_w = abs(xyxy[0].item() - xyxy[2].item())
	bbox_h = abs(xyxy[1].item() - xyxy[3].item())
	x_c = (bbox_left + bbox_w / 2)
	y_c = (bbox_top + bbox_h / 2)
	w = bbox_w
	h = bbox_h
	return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, 
                names=None, color_box=None,offset=(0, 0)):
	for i, box in enumerate(bbox):
		x1, y1, x2, y2 = [int(i) for i in box]
		x1 += offset[0]
		x2 += offset[0]
		y1 += offset[1]
		y2 += offset[1]
		cat = int(categories[i]) if categories is not None else 0
		id = int(identities[i]) if identities is not None else 0
		data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
		label = str(id)

		if color_box:
			color = compute_color_for_labels(id)
			(w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
			cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
			cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
			cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
			[255, 255, 255], 1)
			cv2.circle(img, data, 3, color,-1)
		else:
			(w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
			cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
			cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
			cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
			[255, 255, 255], 1)
			cv2.circle(img, data, 3, (255,191,0),-1)
	return img
#..............................................................................


######

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
		weights = rospy.get_param("~weights")
		# Initialize model
		self.device = select_device(str(rospy.get_param("~device","")))
		# self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
		
		w = str(weights[0] if isinstance(weights, list) else weights)
		model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=self.device)
		stride = int(model.stride.max())  # model stride
		names = model.module.names if hasattr(model, 'module') else model.names  # get class names
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


		# Setting inference size
		self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
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
		
		#.... Initialize SORT .... 
		self.sort_max_age = 5
		self.sort_min_hits = 2
		self.sort_iou_thresh = 0.2
		self.sort_tracker = Sort(max_age=self.sort_max_age, min_hits=self.sort_min_hits, iou_threshold=self.sort_iou_thresh) 
		self.track_color_id = 0
		#......................... 

	def callback(self, data):
		

		"""adapted from yolov5/detect.py"""
		# print(data.header)
		if self.compressed_input:
			im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
		else:
			im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

		im, im0 = self.preprocess(im)
		# print(im.shape)
		# print(img0.shape)
		# print(img.shape)

		# Run inference
		im = torch.from_numpy(im).to(self.device) 
		im = im.half() if self.half else im.float()
		im /= 255
		if len(im.shape) == 3:
			im = im[None]

		pred = self.model(im, augment=False, visualize=False)[0]
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
					# annotator.box_label(xyxy, label, color=colors(c, True))       

				
				### POPULATE THE DETECTION MESSAGE HERE

			#..................USE TRACK FUNCTION....................
			#pass an empty array to sort
			dets_to_sort = np.empty((0,6))

			# NOTE: We send in detected object class too
			for x1,y1,x2,y2,conf,detclass in det:
				dets_to_sort = np.vstack((dets_to_sort, 
									      np.array([x1, y1, x2, y2, 
									                conf, detclass])))
			 
			# Run SORT
			tracked_dets = self.sort_tracker.update(dets_to_sort)
			tracks = self.sort_tracker.getTrackers()


			#loop over tracks
			for track in tracks:
				if False:
					color = compute_color_for_labels(track_color_id)
					[cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
							(int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
							color, thickness=3) for i,_ in  enumerate(track.centroidarr) 
							if i < len(track.centroidarr)-1 ] 
					track_color_id = track_color_id+1
				else:
					[cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
							(int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
							(124, 252, 0), thickness=3) for i,_ in  enumerate(track.centroidarr) 
							if i < len(track.centroidarr)-1 ] 
			if len(tracked_dets)>0:
				bbox_xyxy = tracked_dets[:,:4]
				identities = tracked_dets[:, 8]
				categories = tracked_dets[:, 4]
				draw_boxes(im0, bbox_xyxy, identities, categories)


			# Stream results
			# im0 = annotator.result()

		# Publish prediction
		self.pred_pub.publish(bounding_boxes)

		# Publish & visualize images
		if self.view_image:
			cv2.imshow(str(0), im0)
			cv2.waitKey(1)  # 1 millisecond
		if self.publish_image:
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


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov5Detector()
    
    rospy.spin()
