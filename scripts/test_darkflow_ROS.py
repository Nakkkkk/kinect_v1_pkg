#!/usr/bin/env python
# -*- coding: utf-8 -*-
from darkflow.net.build import TFNet
import cv2
import numpy as np
import colorsys
import random
import time
import rospy
from sensor_msgs.msg import Image
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

class SubscribeImage(object):
	def __init__(self):

 		self.options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 0.2}
#		self.options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu": 0.8}
		self.tfnet = TFNet(self.options)
		self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
							'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
							'dog', 'horse', 'motorbike', 'person', 'pottedplant',
							'sheep', 'sofa', 'train', 'tvmonitor']

		self.num_classes = len(self.class_names)

		# 色リストの作成
		self.hsv_tuples = [(x / 80, 1., 1.) for x in range(80)]
		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), self.hsv_tuples))
		self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),self.colors))
		random.seed(10101)  # Fixed seed for consistent colors across runs.
		random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
		random.seed(None)  # Reset seed to default.
	
		self.periods = []
		self.count = 0
		self.class_num = 0

		rospy.init_node('kinect_v1_image_sub')
		self.bridge = CvBridge()
		rospy.Subscriber('/camera/rgb/image_color', Image, self.callback)
		rospy.spin()


	def callback(self, data):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		(rows,cols,channels) = cv_image.shape
		if cols > 60 and rows > 60 :
			cv2.circle(cv_image, (50,50), 10, 255)

		cv2.imshow("Image window", cv_image)
		self.DarkflowRos(cv_image)


	def DarkflowRos(self, frame):
		# 検出
		self.start = time.time()
		self.items = self.tfnet.return_predict(frame)
		self.count += 1
		self.period = time.time() - self.start
		if self.count % 30 == 0:
			print('FrameRate:' + str(1.0 / (sum(self.periods)/self.count)))

		self.periods.append(self.period)
		for item in self.items:
			self.tlx = item['topleft']['x']
			self.tly = item['topleft']['y']
			self.brx = item['bottomright']['x']
			self.bry = item['bottomright']['y']
			self.label = item['label']
			self.conf = item['confidence']

		# 自信のあるものを表示
		if self.conf > 0.4:
			for i in self.class_names:
				if self.label == i:
					self.class_num = self.class_names.index(i)
					break

		# 検出位置の表示
		cv2.rectangle(frame, (self.tlx, self.tly), (self.brx, self.bry), self.colors[self.class_num], 2)
		self.text = self.label + " " + ('%.2f' % self.conf)
		cv2.putText(frame, self.text, (self.tlx+10, self.tly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[self.class_num], 2)

		# 表示
		cv2.imshow("View", frame)

        k = cv2.waitKey(10)
        if k == ord('q'):
        	cv2.destroyAllWindows()

if __name__ == '__main__':
	try:
		SubscribeImage()
	except rospy.ROSInterruptException:
		pass

