# -*- coding: utf-8 -*-
from darkflow.net.build import TFNet
import cv2
import numpy as np

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)

# 画像の読み込み
img = cv2.imread('raw.jpg')

# 解析を行う
items = tfnet.return_predict(img)
# 検出できたものを確認
print(items)

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']


for item in items:
    # 四角を描くのに必要な情報とラベルを取り出す
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
        cv2.rectangle(img, (tlx, tly), (brx, bry), (200,200,0), 2)
        text = label + " " + ('%.2f' % conf)
        cv2.putText(img, text, (tlx+10, tly-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

# 表示
cv2.imshow("View", img)
cv2.waitKey(0)
# 保存して閉じる
cv2.imwrite('out.jpg', img)
cv2.destroyAllWindows()
