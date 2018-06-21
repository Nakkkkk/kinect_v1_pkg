#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError

class SubscribeImage(object):
    def __init__(self):
        rospy.init_node('kinect_v1_image_sub')
        self.bridge = CvBridge()
        rospy.Subscriber('/camera/rgb/image_color', Image, self.callback)
        rospy.spin()

    def callback(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

#        halfImg = cv2.resize(cv_image, None, fx = 0.5, fy = 0.5)
#        (rows,cols,channels) = cv_image.shape
#        if cols > 60 and rows > 60 :
#            cv2.circle(cv_image, (50,50), 10, 255)

#        cv2.imshow("Image window", halfImg)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

def main():
    try:
        SubscribeImage()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
