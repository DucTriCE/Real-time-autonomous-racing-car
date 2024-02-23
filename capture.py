from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import os

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__))
image_path = ROOT_DIR + '/data'
if __name__ == "__main__":
    try:
        count = 0
        flag = 0
        while True:
            #(180H-y, 320W-x)
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            cv2.imshow('raw_image', raw_image)
            cv2.imshow('segment_image', segment_image)

            AVControl(speed=-10, angle=-10)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                flag = 1
            
            if(count<1000 and flag == 1):
                try:

                    os.chdir(image_path)
                except:
                    os.mkdir(image_path)
                    os.chdir(image_path)
                cv2.imwrite(f'{count}.jpg', raw_image)
                count += 1
                print(count)
            if(count==1000):
                flag = 0
                count = 0


    finally:
        print('closing socket')
        CloseSocket()
        
/*
from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import os

import torch
from matplotlib import pyplot as plt
import numpy as np
import itertools



model = torch.hub.load('ultralytics/yolov5', 'custom', path='./obj_detection/26-10-0215pm-best.pt', force_reload=True, _verbose=False)
model.cuda()
device = torch.device(0)
model.to(device)
model.conf = 0.5


ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__))
image_path = ROOT_DIR + '/data'
if __name__ == "__main__":
    try:
        count = 0
        flag = 0
        font = cv2.FONT_HERSHEY_SIMPLEX  # Font type for text
        font_scale = 0.5  # Font scale
        font_color = (0, 0, 255)  # Font color in BGR
        font_thickness = 1  # Font thickness
        while True:
            #(180H-y, 320W-x)
            state = GetStatus()
            raw_image = GetRaw()
            # cv2.imshow('raw_image', raw_image)
            temp = raw_image.copy()
            count_str = f'Count: {count}'
            cv2.putText(temp, count_str, (10, 30), font, font_scale, font_color, font_thickness)
            cv2.imshow('temp', temp)

            AVControl(speed=-10, angle=-10)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('s'):
                try:

                    os.chdir(image_path)
                except:
                    os.mkdir(image_path)
                    os.chdir(image_path)
                cv2.imwrite(f'{count}.jpg', raw_image)
                count += 1
            
            results = model(raw_image)
            res = np.squeeze(results.render())      #Res se la object detection, ong coi thu shape
            cv2.imshow('object_detection',res)

                # flag = 1
            
            # if(count<1000 and flag == 1):
            #     try:

            #         os.chdir(image_path)
            #     except:
            #         os.mkdir(image_path)
            #         os.chdir(image_path)
            #     cv2.imwrite(f'{count}.jpg', raw_image)
            #     count += 1
            #     print(count)
            # if(count==1000):
            #     flag = 0
            #     count = 0


    finally:
        print('closing socket')
        CloseSocket()
*/
