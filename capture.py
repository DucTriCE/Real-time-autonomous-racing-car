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
