import cv2
import numpy as np
import yaml
import torch
import time
import json
import os
import math
import imutils

from lib.utils.utils import (non_max_suppression, get_cfg, letterbox, scale_boxes,
                            detect, pred_road, gstreamer_pipeline)

from lib.utils.plots import box_label, Colors, show_seg_result, show_det_result
from lib.model.trt import trt_model
from lib.control.UITCar import UITCar

import onnxruntime

# class_dict = ['greenlight', 'left', 'noleft', 'noright', 'nostraight', 'redlight', 'right', 'stop', 'straight', 'yellowlight']

session = onnxruntime.InferenceSession("./models/CNN/CNN-model-020.onnx", providers=['CUDAExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

CURRENT_SPEED = 45
DEFAULT_SPEED = 45
CURVE_SPEED = 30
DETECT_SIGN_SPEED = 20
CHECKPOINT = 50
STOP_TIMER = 0
SIGN = 'None'
LIST_SIGN = []
LIST_LIGHT = []
REDLIGHT_TIMER = 0
CURVE_RIGHT_TIMER = 0
RED_LIGHT_FLAG = 1
CURVE_LEFT_FLAG = 0
CURVE_LEFT_TIMER = 0
SIGN_AREA = 0
TRI_BIP_VAI_LON = 0
STRAIGHT_COUNTER = 0
STOP_FLAG = 0
############################################
# import onnxruntime

# session = onnxruntime.InferenceSession("./models/SEG/model-032.onnx", providers=['CUDAExecutionProvider'])

# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# input_shape = session.get_inputs()[0].shape
# input_type = session.get_inputs()[0].type

# def filter_mask(mask, threshold=200):
#     # Tạo một mask mới với giá trị True cho các pixel lớn hơn 200
#     filtered_mask = mask < threshold

#     # Gán giá trị False cho các pixel không thỏa mãn điều kiện
#     mask[filtered_mask] = 0

#     return mask

# def overlay_mask(original_img, mask):

#     # Resize mask để có kích thước giống ảnh gốc
#     resized_mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

#     # Chuyển đổi mask thành ảnh 3 kênh để so sánh với ảnh gốc
#     mask_color = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

#     # Áp dụng mask lên ảnh gốc
#     overlaid_img = cv2.addWeighted(original_img, 1, mask_color, 0.5, 0)
    
#     # cv2.imshow('Pred', overlaid_img)
#     # cv2.waitKey(0)
#     cv2.imwrite(os.path.join(save_dir, "overlay.jpg"), overlaid_img)
############################################

names = ['greenlight', 'left', 'noleft', 'noright', 'nostraight', 'redlight', 'right', 'stop', 'straight', 'yellowlight']

LANEWIGHT = 55            # Độ rộng đường (pixel)
IMAGESHAPE = [160, 80]      # Kích thước ảnh Input model 

def most_frequent(List):
    return max(set(List), key = List.count)
# def left_curve(cap):
#     _ = cap.read()
def AngCal(image, cap, car):
    global DEFAULT_SPEED
    global CURRENT_SPEED
    global CURVE_SPEED
    global DETECT_SIGN_SPEED
    global CHECKPOINT
    global STOP_TIMER
    global SIGN
    global LIST_SIGN
    global LIST_LIGHT
    global REDLIGHT_TIMER
    global CURVE_RIGHT_TIMER
    global RED_LIGHT_FLAG
    global CURVE_LEFT_FLAG
    global CURVE_LEFT_TIMER
    global SIGN_AREA
    global TRI_BIP_VAI_LON
    global STRAIGHT_COUNTER
    global STOP_FLAG
    # print("hello######################################")
    # if(SIGN !="None"):
        # 0: greenlight
        # 1: left
        # 2: noleft
        # 3: noright
        # 4: nostraight
        # 5: redlight
        # 6: right
        # 7: stop
        # 8: straight
        # 9: yellowlight

        # if (SIGN == "greenlight"):
        #     LIST_SIGN.append(0)
        # if (SIGN == "left"):
        #     LIST_SIGN.append(1)
        # if (SIGN == "noleft"):
        #     LIST_SIGN.append(2)
        # if (SIGN == "noright"):
        #     LIST_SIGN.append(3)
        # if (SIGN == "nostraight"): 
        #     LIST_SIGN.append(4)
        # if (SIGN == "redlight"): 
        #     LIST_SIGN.append(5)
        # if (SIGN == "right"): 
        #     LIST_SIGN.append(6)
        # if (SIGN == "stop"): 
        #     LIST_SIGN.append(7)
        # if (SIGN == "straight"): 
        #     LIST_SIGN.append(8)
        # if (SIGN == "yellowlight"): 
        #     LIST_SIGN.append(9)
    
    if (len(LIST_LIGHT)>3):
        # right_y, left_y = 0, 0
        # line_column_left = image[:, 60]
        # for x, y in enumerate(reversed(line_column_left)):
        #     if y ==255:
        #         left_y =  x
        # line_column_right = image[:, 100]
        # for x, y in enumerate(reversed(line_column_right)):
        #     if y ==255:
        #         right_y =  x
        if (SIGN_AREA>2000):
            if(most_frequent(LIST_LIGHT)==5 or most_frequent(LIST_LIGHT)==9): # redlight
                # REDLIGHT_TIMER = REDLIGHT_TIMER+1
                RED_LIGHT_FLAG = 1
                LIST_LIGHT = []
                CURRENT_SPEED = 0
                car.setSpeed_cm(-3)
                return 0, CURRENT_SPEED
                # return 0, CURRENT_SPEED
        if(len(LIST_LIGHT)>3 and (most_frequent(LIST_LIGHT)==0)): #or REDLIGHT_TIMER>100)):
            CURRENT_SPEED = DEFAULT_SPEED
            CHECKPOINT=50
            REDLIGHT_TIMER = 0
            LIST_LIGHT = []
            RED_LIGHT_FLAG = 0
    if(RED_LIGHT_FLAG==1):
        CURRENT_SPEED = 0
        car.setSpeed_cm(-3)
        return 0, CURRENT_SPEED
            # car.setAngle(0)
            # CURRENT_SPEED = 0
            # REDLIGHT_TIMER = REDLIGHT_TIMER + 1
        
    if (len(LIST_SIGN)>2):
        if(most_frequent(LIST_SIGN)==7):
            CURRENT_SPEED = 0
            # if(STOP_FLAG==0):
            #     STOP_FLAG=1
            #     return 0, -58
            car.setSpeed_cm(-3)
            return 0, CURRENT_SPEED
        right_y, left_y = 0, 0
        line_column_left = image[:, 70]
        for x, y in enumerate(reversed(line_column_left)):
            if y ==255:
                left_y =  x
        line_column_right = image[:, 90]
        for x, y in enumerate(reversed(line_column_right)):
            if y ==255:
                right_y =  x
        if(most_frequent(LIST_SIGN)==4 or most_frequent(LIST_SIGN)==1 or most_frequent(LIST_SIGN)==3):
            # if (left_y>44 and right_y>44):
            print("##########Processing curve left###########")
            if(SIGN_AREA>3500 and most_frequent(LIST_SIGN)==1):
                car.setSpeed_cm(40)
                car.setAngle(10) #11.9v
                for i in range (1, 75):
                    _ = cap.read()
                    print(i)
                LIST_SIGN = []
                CURVE_LEFT_FLAG = 1
            if((len(LIST_SIGN)>2) and SIGN_AREA>5500 and (most_frequent(LIST_SIGN)==4 or most_frequent(LIST_SIGN)==3)):
                car.setSpeed_cm(40)
                car.setAngle(14) #11.9v
                for i in range (1, 70):
                    _ = cap.read()
                    print(i)
                LIST_SIGN = []
                CURVE_LEFT_FLAG = 1
            
            if(CURVE_LEFT_FLAG != 1):
                CHECKPOINT = 50
                line_row = image[CHECKPOINT, :]
                max_x = 0
                for x, y in enumerate(reversed(line_row)):
                    if y == 255:
                        max_x = 160-x
                        break
                if(max_x==160):
                    center_row = 90
                else:
                    center_row = max_x-40
                image = cv2.circle(image, (center_row, CHECKPOINT), 1, 40, 2)
                h, w = 80, 160
                x0, y0 = int(w/2), h
                x1, y1 = center_row, CHECKPOINT
                value = (x1-x0)/(y0-y1)

                angle = math.degrees(math.atan(value))
                speed = CURRENT_SPEED
                CURVE_RIGHT_TIMER = CURVE_RIGHT_TIMER + 1
                return -angle/1.4, speed
        
                # car.setSpeed_cm(40)
                # car.setAngle(12) #11.9v
                # # for i in range (1, 70):
                # #     _ = cap.read()
                # #     print(i)
                # CHECKPOINT = 70
                # LIST_SIGN = []
                # CURVE_LEFT_FLAG = 1

        if(len(LIST_SIGN)>3 and  most_frequent(LIST_SIGN)==8):
            if(CURVE_RIGHT_TIMER<70):
                print("###########Processing straight#############")
                CHECKPOINT = 50
                line_row = image[CHECKPOINT, :]
                max_x = 0
                for x, y in enumerate(reversed(line_row)):
                    if y == 255:
                        max_x = 160-x
                        break
                if(max_x==160):
                    center_row = 90
                else:
                    center_row = max_x-30
                image = cv2.circle(image, (center_row, CHECKPOINT), 1, 40, 2)
                h, w = 80, 160
                x0, y0 = int(w/2), h
                x1, y1 = center_row, CHECKPOINT
                value = (x1-x0)/(y0-y1)

                angle = math.degrees(math.atan(value))
                speed = CURRENT_SPEED
                CURVE_RIGHT_TIMER = CURVE_RIGHT_TIMER + 1
                return -angle/1.5, speed
            if(CURVE_RIGHT_TIMER>=70):
                CURVE_RIGHT_TIMER=0
                LIST_SIGN = []
                # STRAIGHT_COUNTER = STRAIGHT_COUNTER + 1
                # if(STRAIGHT_COUNTER==2):
                #     TRI_BIP_VAI_LON = 1
                # if(STRAIGHT_COUNTER==3):
                #     STRAIGHT_COUNTER = 1
        if(len(LIST_SIGN)>3 and (most_frequent(LIST_SIGN)==6 or most_frequent(LIST_SIGN)==2)):
            if(CURVE_RIGHT_TIMER<60):
                print("###########Processing curve right#############")
                CHECKPOINT = 50
                line_row = image[CHECKPOINT, :]
                max_x = 0
                for x, y in enumerate(reversed(line_row)):
                    if y == 255:
                        max_x = 160-x
                        break
                if(max_x==160):
                    center_row = 115
                else:
                    center_row = max_x-28
                image = cv2.circle(image, (center_row, CHECKPOINT), 1, 40, 2)
                h, w = 80, 160
                x0, y0 = int(w/2), h
                x1, y1 = center_row, CHECKPOINT
                value = (x1-x0)/(y0-y1)

                angle = math.degrees(math.atan(value))
                speed = CURRENT_SPEED
                CURVE_RIGHT_TIMER = CURVE_RIGHT_TIMER + 1
                return -angle/1.4, speed
            if(CURVE_RIGHT_TIMER>=60):
                if(TRI_BIP_VAI_LON==0):
                    TRI_BIP_VAI_LON = 1
                else:
                    TRI_BIP_VAI_LON = 0
                CURVE_RIGHT_TIMER=0
                LIST_SIGN = []

    h, w = 80, 160

    line_row = image[CHECKPOINT, :]
    # center = image[h-5, :]
    
    # flag = True
    # center_min_x = 0
    # center_max_x = 0
    
    # for x, y in enumerate(center):
    #     if y == 255 and flag:
    #         flag = False
    #         center_min_x = x
    #     elif y == 255:
    #         center_max_x = x
            
    # center_segment = int((center_max_x+center_min_x)/2)
    
    # flag = True
    min_x = 0
    max_x = 0

    for x, y in enumerate(line_row):
        if y == 255:
            min_x = x
            break
    for x, y in enumerate(reversed(line_row)):
        if y == 255:
            max_x = w-x
            break
    if (CURVE_LEFT_FLAG == 1):
        center_row = max_x -40
        CURVE_LEFT_TIMER = CURVE_LEFT_TIMER + 1
    if (CURVE_LEFT_TIMER>20):
        CURVE_LEFT_TIMER= 0
        CURVE_LEFT_FLAG=0
    if (CURVE_LEFT_FLAG == 0):
        center_row = int((max_x+min_x)/2)

    print("Center row: ", center_row)
    # gray = cv2.circle(gray, (center_row, CHECKPOINT), 1, 90, 2)
    # cv2.imshow('test', gray)
    
    image = cv2.circle(image, (center_row, CHECKPOINT), 1, 40, 2)

    x0, y0 = int(w/2), h
    x1, y1 = center_row, CHECKPOINT
    
    # image=cv2.line(image,(x1, y1),(x0, y0),(0,0,0),10)

    value = (x1-x0)/(y0-y1)

    angle = math.degrees(math.atan(value))

    # print(steering)
    

	# _lineRow = image[CHECKPOINT, :] 
	# count = 0
	# sumCenter = 0
	# centerArg = int(IMAGESHAPE[0]/2)
	# minx=0
	# maxx=0
	# first_flag=True
	# for x, y in enumerate(_lineRow):
	# 	if y == 255 and first_flag:
	# 		first_flag=False
	# 		minx=x
	# 	elif y == 255:
	# 		maxx=x
	 
	# # centerArg = int(sumCenter/count)
	# centerArg=int((minx+maxx)//2)
	# count=maxx-minx

	# # print(minx,maxx,centerArg)
	# # print(centerArg, count)

    # if (count < LANEWIGHT):
    #     if (centerArg < int(IMAGESHAPE[0]/2)):
    #         centerArg -= LANEWIGHT - count
    #     else:
    #         centerArg += LANEWIGHT - count

	# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	# _steering = math.degrees(math.atan((centerArg - int(IMAGESHAPE[0]/2))/(IMAGESHAPE[1]-CHECKPOINT)))
	# # print(_steering,"----------",count)
	# image=cv2.line(image,(centerArg,CHECKPOINT),(int(IMAGESHAPE[0]/2),IMAGESHAPE[1]),(255,0,0),1)
    speed = CURRENT_SPEED
    if (TRI_BIP_VAI_LON==1):
        speed = 30
        return -angle/1.0, speed
    return -angle/1.5, speed


def control(car, speed, angle):
    car.setSpeed_cm(speed)
    car.setAngle(angle)


if __name__ == "__main__":
    # # Motor init
    Car = UITCar()
    # Car.getSpeed_cm
    # Car.setMotorMode(0)
    # Car.setAngle()
    # Car.setSpeed_cm()
    # Car.getMotor_Current
    # Setting Camera
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    assert cap.isOpened(), "Camera failed"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    colors = Colors()
    engine, torchscript, imgsize_yolo, imgsize_seg, visualize, save_dir = get_cfg('lib/cfg/cfg.yaml')


    # Loading YOLOv5
    yolo = trt_model(engine)
    yolo = yolo.half()
    yolo.warmup(imgsz=(1, 3, *imgsize_yolo))  # warmup

    
    # Loading Segmentation model
    print(f'Loading {torchscript} for TorchScript inference...')
    extra_files = {'config.txt': ''}  # model metadata
    model_seg = torch.jit.load(torchscript, _extra_files=extra_files, map_location=device)
    model_seg = model_seg.to(device).float()    

    # i = 0
    # count = 0

    while 1:
        # if(STOP_TIMER < 20000):
        #     STOP_TIMER = STOP_TIMER +1
        # else:
        #     Car.setMotorMode(0)
        #     Car.setAngle(0)
        #     Car.setSpeed_cm(0)
        #     break
        # start = time.time()
        ret, frame = cap.read()
        assert ret, "Failed to read camera"

        mask_pred = pred_road(model_seg, frame, imgsize_seg, device)
        # dilations to remove any small regions of noise
        mask_pred = cv2.erode(mask_pred, None, iterations=8)
        # cv2.imshow("image5", mask_pred)
        mask_pred = cv2.dilate(mask_pred, None, iterations=8)
        
        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(mask_pred.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        # print(cnts)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
        else:
            Car.setSpeed_cm(20)
            Car.setAngle(0)
            continue
        # hull = cv2.convexHull(c)

        # Extract the vertices of the convex hull
        # extreme_points = hull.reshape((-1, 2))

        # print(c)
        # determine the most extreme points along the contour

        mask = np.zeros_like(mask_pred)

        # Draw the contour 'c' on the mask with white color
        cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Draw the extreme points on the image
        # for point in extreme_points:
        #     point = tuple(point)
        #     cv2.circle(frame, point, 4, (0, 0, 255), -1)

        # Use the mask to extract the region of interest (ROI) from the original image
        result = cv2.bitwise_and(mask_pred, mask)
        ############

        cv2.imwrite(os.path.join(save_dir, "binary.jpg"), result)
        if visualize:
            show_seg_result(frame, mask_pred, imgsize_seg, os.path.join(save_dir, "mask.jpg"))

        # img = frame[240:,:]
        # img = cv2.resize(img, (640, 400))
        # input_data = np.expand_dims(img, axis=0).astype(np.float32)
        # output = session.run([output_name], {input_name: input_data})
        # segmentation_result = output[0]
        # segmentation_result = (segmentation_result - np.min(segmentation_result)) / (np.max(segmentation_result) - np.min(segmentation_result)) * 255
        # segment_image = segmentation_result.astype(np.uint8)
        # res = filter_mask(segment_image[0], 200)
        # cv2.imwrite(os.path.join(save_dir, "mask.jpg"), segment_image[0])
        # overlay_mask(img, res)
        # cv2.imshow('segment', segment_image[0])


        #This is for YOLO
        pred, im = detect(yolo, frame, imgsize_yolo, device)
        
        for _, det in enumerate(pred):  # per image
            _ = cap.read()
            print("####################################################")
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, _ in reversed(det):
                    top, left, bottom, right = xyxy
                    top, left, bottom, right = top.cpu().numpy().astype(int), left.cpu().numpy().astype(int), bottom.cpu().numpy().astype(int), right.cpu().numpy().astype(int)
                    SIGN_AREA = int(abs(top-bottom)*abs(left-right))
                    cropped_box = frame[left:right, top:bottom]
                    w, h, c = cropped_box.shape
                    if (w>0 and h>0):
                        cropped_box = cv2.resize(cropped_box, (50, 50))
                        tmp = cropped_box.copy()
                        cropped_box = np.expand_dims(cropped_box, axis=0)
                        cropped_box = cropped_box.astype('float32')/255.0
                        traffic_sign = session.run([output_name], {input_name: cropped_box})[0]
                        # if (i%40==0):
                        #     cv2.imwrite(os.path.join(save_dir, f"cropped_box/{count}.jpg"), tmp)
                        #     count+=1
                        # i+=1
                        conf = max(traffic_sign[0])
                        if(conf>0.7): 
                            cls=np.argmax(traffic_sign[0])
                            print(names[cls])
                            if (names[cls] == "greenlight"):
                                LIST_LIGHT.append(0)
                            if (names[cls] == "left"):
                                LIST_SIGN.append(1)
                            if (names[cls] == "noleft"):
                                LIST_SIGN.append(2)
                            if (names[cls] == "noright"):
                                LIST_SIGN.append(3)
                            if (names[cls] == "nostraight"): 
                                LIST_SIGN.append(4)
                            if (names[cls] == "redlight"): 
                                LIST_LIGHT.append(5)
                            if (names[cls] == "right"): 
                                LIST_SIGN.append(6)
                            if (names[cls] == "stop"): 
                                LIST_SIGN.append(7)
                            if (names[cls] == "straight"): 
                                LIST_SIGN.append(8)
                            if (names[cls] == "yellowlight"): 
                                LIST_LIGHT.append(9)
                            if visualize:
                                show_det_result(frame, xyxy, cls, names, conf, colors, os.path.join(save_dir, "det.jpg"))
                        else:
                            cv2.imwrite(os.path.join(save_dir, "det.jpg"), frame)
                    else:
                        cv2.imwrite(os.path.join(save_dir, "det.jpg"), frame)
            elif visualize:
                # cv2.imshow('obj', frame)
                # cv2.waitKey(1)
                cv2.imwrite(os.path.join(save_dir, "det.jpg"), frame)
                SIGN = 'None'
        
        ############### THIS IS WHERE CONTROL ALGORITHM BEGIN #############
        current_speed = Car.getAngle
        current_angle = Car.getSpeed_cm
        angle, speed = AngCal(image=result, cap=cap, car=Car)
        control(car=Car, speed=speed, angle=angle)
        # print("List sign: ", LIST_SIGN)
        # print("List light: ", LIST_LIGHT)
        # print("Current Speed: ", speed)
        # print("Current Angle: ", angle)
        # # print("Detect: ", Detect_FLAG)
        # print("CheckPoint: ", CHECKPOINT)
        # print("Curve right timer: ", CURVE_RIGHT_TIMER)
        # print("Curve left timer: ", CURVE_LEFT_TIMER)
        # print("Curve left flag: ", CURVE_LEFT_FLAG)
        # print("Sign area: ", SIGN_AREA)
        # ################ CONTROL ALGORITHM END HERE #######################
        # print("Tri bip vai lon: ", TRI_BIP_VAI_LON)
        # print("Red light flag: ", RED_LIGHT_FLAG)
        # print(1//(time.time()-start))



        


