# -*- coding: utf-8 -*-
from darkflow.net.build import TFNet
import cv2
import numpy as np
import colorsys
import random
import time

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "GPU": 1.0}
#options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1, "gpu": 0.8}
tfnet = TFNet(options)

# 動画の読み込み
cap = cv2.VideoCapture(0)
'''
# 動画保存の設定
fps = 30
size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_tiny_yolo_voc.mp4', fourcc, fps,size)
out = cv2.VideoWriter('output_yolo.mp4', fourcc, fps,size)
'''
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']

num_classes = len(class_names)

# 色リストの作成
hsv_tuples = [(x / 80, 1., 1.) for x in range(80)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

periods = []
count = 0
while True:
    # フレームを取得
    ret, frame = cap.read()
    if not ret:
        break
    # 検出
    start = time.time()
    items = tfnet.return_predict(frame)
    count += 1
    period = time.time() - start
    if count % 30 == 0:
        print('FrameRate:' + str(1.0 / (sum(periods)/count)))
    
    periods.append(period)
    for item in items:
        tlx = item['topleft']['x']
        tly = item['topleft']['y']
        brx = item['bottomright']['x']
        bry = item['bottomright']['y']
        label = item['label']
        conf = item['confidence']

        # 自信のあるものを表示
        if conf > 0.4:

            for i in class_names:
                if label == i:
                    class_num = class_names.index(i)
                    break

            # 検出位置の表示
            cv2.rectangle(frame, (tlx, tly), (brx, bry), colors[class_num], 2)
            text = label + " " + ('%.2f' % conf)
            cv2.putText(frame, text, (tlx+10, tly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_num], 2)

    # 表示
    cv2.imshow("View", frame)
    # 保存
#    out.write(frame)
    # qで終了
    k = cv2.waitKey(10);
    if k == ord('q'):  break;

cap.release()
#out.release()
cv2.destroyAllWindows()
