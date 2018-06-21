#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy
import time
import cv2
from sensor_msgs.msg import Image
from matplotlib import pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import colorsys
import random


class SubscribePointCloud(object):
    def __init__(self):
        rospy.init_node('kinect_v1_depth_sub')
        rospy.Subscriber('/camera/depth/points', PointCloud2, self.callback)
#        self.bridge = CvBridge()
#        rospy.Subscriber('/camera/rgb/image_color', Image, self.callback)

        rospy.spin()

#    def callback(self, point_cloud):
    def callback(self, data):

#        print "data = " + str(data.height) + str(data.width)
        resolution = (data.height, data.width)

        # 3D position for each pixel
        img = numpy.fromstring(data.data, numpy.float32)
        halfImg = cv2.resize(img, None, fx = 0.5, fy = 0.5)
        half_resolution = halfImg.shape[:2]

#        cloud_points_x = []
#        cloud_points_y = []
        cloud_points_z = []
        for p in pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=False):
#            cloud_points_x.append(p[0])
#            cloud_points_y.append(p[1])
            cloud_points_z.append(p[2])

#        x_points = numpy.array(cloud_points_x, dtype=numpy.float32)
#        y_points = numpy.array(cloud_points_y, dtype=numpy.float32)
        z_points = numpy.array(cloud_points_z, dtype=numpy.float32)

#        print "len = " + str(len(z_points))

        z = z_points.reshape(resolution)
#        z_max = numpy.argmin(z)
#        z_max_h = z_max / data.width
#        z_max_w = z_max % data.width

#        cv2.rectangle(cv_image, (z_max_w, z_max_h), (60, 60), (0, 255, 0))

#        cv2.imshow("test",cv_image)
#        k = cv2.waitKey(10)
#        if k == ord('q'):
#            cv2.destroyAllWindows()

#        z = z_points.reshape(resolution)
        print z[data.height/2 , data.width/2]
#        print z[z_max_h , z_max_w]
#        print "z max = " + str(z_max) + " , (h,w) = " + str(z_max_h) + " , " + str(z_max_w)

        '''
        for point in pc2.read_points(point_cloud):
            rospy.logwarn("x, y, z: %.1f, %.1f, %.1f" % (point[0], point[1], point[2]))
            rospy.logwarn("my field 1: %f" % (point[4]))
            rospy.logwarn("my field 2: %f" % (point[5]))
        '''

def main():
    try:
        SubscribePointCloud()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
