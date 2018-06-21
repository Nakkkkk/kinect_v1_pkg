#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from darkflow.net.build import TFNet
import colorsys
import random


class SubscribeImage():
	def __init__(self):
		rospy.init_node('kinect_v1_image_sub')
		self.bridge = CvBridge()
		rospy.Subscriber('/camera/rgb/image_color', Image, self.callback)
		rospy.spin()

		self.options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "GPU": 1.0}
		#options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu": 0.8}
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

	def callback(self, data):

		try:
			frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		(rows,cols,channels) = frame.shape
		if cols > 60 and rows > 60 :
			cv2.circle(frame, (50,50), 10, 255)

#		darkros(cv_image,tfnet,class_name,)
#        cv2.imshow("Image window", cv_image)
#        cv2.waitKey(3)

#def darkros(frame):
		# フレームを取得
#		ret, frame = cap.read()
#		if not ret:
#			break
		# 検出
		start = time.time()
		items = self.tfnet.return_predict(frame)
		self.count += 1
		period = time.time() - start
		if count % 30 == 0:
			print('FrameRate:' + str(1.0 / (sum(self.periods)/self.count)))

		self.periods.append(period)
		for item in items:
			tlx = item['topleft']['x']
			tly = item['topleft']['y']
			brx = item['bottomright']['x']
			bry = item['bottomright']['y']
			label = item['label']
			conf = item['confidence']

			# 自信のあるものを表示
			if conf > 0.4:

				for i in self.class_names:
					if label == i:
						class_num = self.class_names.index(i)
						break

			# 検出位置の表示
			cv2.rectangle(frame, (tlx, tly), (brx, bry), self.colors[class_num], 2)
			text = label + " " + ('%.2f' % conf)
			cv2.putText(frame, text, (tlx+10, tly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[class_num], 2)

		# 表示
		cv2.imshow("View", frame)
		# 保存
		#out.write(frame)
		# qで終了
		k = cv2.waitKey(10);
		if k == ord('q'):
			cv2.destroyAllWindows()


def main():
    try:
        SubscribeImage()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':

    main()

