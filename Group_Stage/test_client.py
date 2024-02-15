from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import numpy as np 
import os
import cv2
import math
import onnxruntime

import torch
from matplotlib import pyplot as plt
import numpy as np
import itertools

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

YOLO_model = YOLO('./model_obj_detection/Koi_v8.pt')


session = onnxruntime.InferenceSession("./model/model044.onnx")
session1 = onnxruntime.InferenceSession("model_obj_detection/CNN-model-200.onnx")

class_dict = {0: 'left', 1:'noleft', 2:'noright', 3:'right', 4:'stop', 5:'straight'}


input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

input_name1 = session1.get_inputs()[0].name
output_name1 = session1.get_outputs()[0].name

TRI_BIP_VAI_LON = 0

DEFAULT_SPEED = 40
CURVE_SPEED = 30
DETECT_SIGN_SPEED = 20
CHECKPOINT = 120
REFERENCE_POINT_OF_STRAIGHT = 120
REFERENCE_POINT_OF_CURVE = 120
REFERENCE_POINT_OF_EXTREME_CURVE = 120
REFERENCE_POINT_OF_SPECIAL_CURVE = 100
NUM_OF_SAMPLE_ROW = 10
TIMER = 0
LOOP_COUNTER = 5
temp_READY_FLAG = 0
READY_FLAG = 0
STOP_FLAG = 0
DETECT_SIGN = 0 # 0: none, 1: straight, 2:left, 3: right, 4:stop
MODE = 0 # mode 0 is normal condition, mode 1 is detect sign
SKY_HEIGHT = 85 #WITDH-(NUM_OF_SAMPLE_ROW-1)*10 = 176-9*10
LIST_SIGN = []

# This is values of Detect_FLAG
CURVE = 0
EXTREME_CURVE = 1
STRAIGHT = 2
CROSS_ROAD = 3
SPECIAL_CURVE = 4

# flag to dectect road ahead
Detect_FLAG = 2

def most_frequent(List):
    return max(set(List), key = List.count)

def LinePassingTwoPoint(x1, y1, x2, y2):
    # y=ax+b
    if (x1 == x2):
        a = 0
    else:
        a = float((y1-y2)/(x1-x2))
    b = float(y1-a*x1)
    return a, b

def IntersecTwoLine(a1, b1, a2, b2):
    if (a1 == a2):
        x = 160
    else:
        x = float((b2-b1)/(a1-a2))
    y = int(a1*x + b1)
    return int(x), y

def IntersecFromFourPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    a1, b1 = LinePassingTwoPoint(x1, y1, x2, y2)
    a2, b2 = LinePassingTwoPoint(x3, y3 ,x4, y4)
    x, y = IntersecTwoLine(a1, b1, a2, b2)
    return x, y

def AngCal(image ,current_speed, class_label):
    global Detect_FLAG
    global CHECKPOINT
    global REFERENCE_POINT_OF_SPECIAL_CURVE
    global TIMER
    global MODE
    global ARR_SIGN
    global DETECT_SIGN
    global LOOP_COUNTER
    global temp_READY_FLAG
    global READY_FLAG
    global TRI_BIP_VAI_LON
    global STOP_FLAG
    speed = DEFAULT_SPEED

    h, w, _  = image.shape
    Min_Max_of_Sample_Row = []
    Avg_of_each_Sample_row = []
    Avg_of_each_Sample_row_detail = []
    STRAIGHT_FLAG = 0
    CROSS_FLAG = 0

    if(class_label!="none"):
        # 0: left
        # 1: no_left
        # 2: no_right
        # 3: right
        # 4: stop
        # 5: straight

        if (class_label == "left"):
            LIST_SIGN.append(0)
        if (class_label == "noleft"):
            LIST_SIGN.append(1)
        if (class_label == "noright"):
            LIST_SIGN.append(2)
        if (class_label == "right"):
            LIST_SIGN.append(3)
        if (class_label == "stop"): 
            LIST_SIGN.append(4)
        if (class_label == "stop" and most_frequent(LIST_SIGN) == 4 and len(LIST_SIGN)>2):
            STOP_FLAG=1
            if(current_speed>20):
                return 0,-90
            else:
                return 0,0
        if (class_label == "straight"):
            LIST_SIGN.append(5)
        if (len(LIST_SIGN)>5 and class_label=="noleft"):
            MODE = 1
        if (len(LIST_SIGN)>10):
            MODE = 1
    if(STOP_FLAG):
        return 0,0
    if(MODE == 1):
        speed = DETECT_SIGN_SPEED
        if(TIMER<83):
            if (LOOP_COUNTER == 0):
                READY_FLAG = 0
            TIMER = TIMER + 1
            CHECKPOINT = 135
            Min_Max_of_Sample_Row_SIGN = []

            right_y, right_y1, left_y, right_y0 = 0, 0, 0, 0
            line_column_left = image[:, 0]
            for x, y in enumerate(reversed(line_column_left)):
                if y > 210:
                    left_y =  175 - x

            line_column_left1 = image[:, 35]
            for x, y in enumerate(reversed(line_column_left1)):
                if y > 210:
                    left_y1 =  175 - x

            line_column_right = image[:, 319]
            for x, y in enumerate(reversed(line_column_right)):
                if y > 210:
                    right_y = 175 - x

            line_column_right1 = image[:, 285]
            for x, y in enumerate(reversed(line_column_right1)):
                if y > 210:
                    right_y1 = 175 - x

            
            if (right_y<104 or left_y<104):
                temp_READY_FLAG = 1
            if (temp_READY_FLAG and (right_y>107 and left_y>107) and (abs(left_y-left_y1)<8 or abs(right_y-right_y1)<8)):
                READY_FLAG = 1


            if (READY_FLAG == 1):
                if (DETECT_SIGN == 0):
                    LOOP_COUNTER = LOOP_COUNTER + 1
                    most_appear = most_frequent(LIST_SIGN)
                    if(most_appear == 0 or most_appear == 2):
                        DETECT_SIGN = 2
                    if(most_appear == 1 or most_appear == 3):
                        DETECT_SIGN = 3
                    if(most_appear == 4):
                        DETECT_SIGN = 4
                    if(most_appear == 5):
                        DETECT_SIGN == 1
                if (DETECT_SIGN != 0):
                    if (DETECT_SIGN == 1):
                        CHECKPOINT = 100
                        speed = 40
                        LOOP_COUNTER = LOOP_COUNTER + 1
                        if (LOOP_COUNTER>7):
                            TIMER = 0
                            MODE = 0
                            LIST_SIGN.clear()
                            DETECT_SIGN = 0
                            LOOP_COUNTER = 0
                            temp_READY_FLAG = 0
                            READY_FLAG = 0
                    if (DETECT_SIGN == 2):
                        LOOP_COUNTER = LOOP_COUNTER + 1
                        if (LOOP_COUNTER>14):
                            TIMER = 0
                            MODE = 0
                            LIST_SIGN.clear()
                            DETECT_SIGN = 0
                            LOOP_COUNTER = 0
                            temp_READY_FLAG = 0
                            READY_FLAG = 0
                        else:
                            return -25,20
                    if (DETECT_SIGN == 3):
                        LOOP_COUNTER = LOOP_COUNTER + 1
                        if (LOOP_COUNTER>15):
                            TIMER = 0
                            MODE = 0
                            LIST_SIGN.clear()
                            DETECT_SIGN = 0
                            LOOP_COUNTER = 0
                            temp_READY_FLAG = 0
                            READY_FLAG = 0
                        else:
                            return 25,20
                    if (DETECT_SIGN == 4):
                        CHECKPOINT = 125
                        speed = 0



                    line_row = image[CHECKPOINT, :]
                    min_x = 0
                    max_x = 320
                
                    for x, y in enumerate(line_row):
                        if y > 210:
                            min_x = x
                            break

                    for x, y in enumerate(reversed(line_row)):
                        if y > 210:
                            max_x = 320-x
                            break
                    
                    center_row = int((max_x+min_x)/2)
                    x0, y0 = int(w/2), h
                    x1, y1 = center_row, CHECKPOINT


                    value = (x1-x0)/(y0-y1)

                    angle = math.degrees(math.atan(value))
                    image = cv2.circle(image, (x1, CHECKPOINT), 10,40)
                    return angle/7, speed
                    
                # return 
            if (READY_FLAG==0):
                
                line_row = image[CHECKPOINT, :]
                min_x = 0
                max_x = 0

                for x, y in enumerate(line_row):
                    if y > 210:
                        min_x = x
                        break

                for x, y in enumerate(reversed(line_row)):
                    if y > 210:
                        max_x = 320-x
                        break
                center_row = int((max_x+min_x)/2)
                image = cv2.circle(image, (center_row, CHECKPOINT), 1, 40, 2)
                
                x0, y0 = int(w/2), h
                x1, y1 = center_row, CHECKPOINT

                value = (x1-x0)/(y0-y1)

                angle = math.degrees(math.atan(value))

                if (current_speed > speed):
                    speed = 0
                return angle/6, speed

        # escape after 100 loop
        else:
            TIMER = 0
            MODE = 0
            LIST_SIGN.clear()
            DETECT_SIGN = 0
            LOOP_COUNTER = 0
            temp_READY_FLAG = 0
            READY_FLAG = 0

    for i in range(1, 5):
        line_row = image[SKY_HEIGHT+5+i*5, :]
        min_x = 0
        max_x = 320
    
        for x, y in enumerate(line_row):
            if y > 210:
                min_x = x
                break

        for x, y in enumerate(reversed(line_row)):
            if y > 210:
                max_x = 320-x
                break
        temp_list=[]
        temp_list.append(min_x)
        temp_list.append(max_x)
        #Draw middle points
        image = cv2.circle(image, (int((min_x + max_x)/2), SKY_HEIGHT+5+i*5), 1, 90)
        Avg_of_each_Sample_row_detail.append(int((min_x + max_x)/2))

    for i in range(NUM_OF_SAMPLE_ROW):
        line_row = image[SKY_HEIGHT+i*10, :]
        min_x = 0
        max_x = 320
    
        for x, y in enumerate(line_row):
            if y > 210:
                min_x = x
                break

        for x, y in enumerate(reversed(line_row)):
            if y > 210:
                max_x = 320-x
                break
        temp_list=[]
        temp_list.append(min_x)
        temp_list.append(max_x)
        Min_Max_of_Sample_Row.append(temp_list)
        #Draw middle points
        image = cv2.circle(image, (int((min_x + max_x)/2), SKY_HEIGHT+i*10), 1, 90)
        Avg_of_each_Sample_row.append(int((min_x + max_x)/2))

    #detect straight
    for i in range (1, 3):
        min_x, max_x = Min_Max_of_Sample_Row[i]
        if (min_x>0 and max_x<320
        and Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1]>0 and Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1]<14
        and Avg_of_each_Sample_row[i+1]-Avg_of_each_Sample_row[i+2]>0 and Avg_of_each_Sample_row[i+1]-Avg_of_each_Sample_row[i+2]<14
        and Avg_of_each_Sample_row[i+2]-Avg_of_each_Sample_row[i+3]>0 and Avg_of_each_Sample_row[i+2]-Avg_of_each_Sample_row[i+3]<14
        and abs(Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1])-abs(Avg_of_each_Sample_row[i+1]-Avg_of_each_Sample_row[i+2])<8
        and abs(Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1])-abs(Avg_of_each_Sample_row[i+2]-Avg_of_each_Sample_row[i+3])<8):
            STRAIGHT_FLAG=1
            break
        if (min_x>0 and max_x<320
        and Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1]<0 and Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1]>-14
        and Avg_of_each_Sample_row[i+1]-Avg_of_each_Sample_row[i+2]<0 and Avg_of_each_Sample_row[i+1]-Avg_of_each_Sample_row[i+2]>-14
        and Avg_of_each_Sample_row[i+2]-Avg_of_each_Sample_row[i+3]<0 and Avg_of_each_Sample_row[i+2]-Avg_of_each_Sample_row[i+3]>-14
        and abs(Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1])-abs(Avg_of_each_Sample_row[i+1]-Avg_of_each_Sample_row[i+2])<8
        and abs(Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1])-abs(Avg_of_each_Sample_row[i+2]-Avg_of_each_Sample_row[i+3])<8):
            STRAIGHT_FLAG=1
            break
        if (min_x>0 and max_x<320
        and abs(Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1])-abs(Avg_of_each_Sample_row[i+1]-Avg_of_each_Sample_row[i+2])<8
        and abs(Avg_of_each_Sample_row[i]-Avg_of_each_Sample_row[i+1])-abs(Avg_of_each_Sample_row[i+2]-Avg_of_each_Sample_row[i+3])<8):
            if (Detect_FLAG!=CROSS_ROAD):
                STRAIGHT_FLAG=1
                break

    temp_straight_flag=1
    for i in range(NUM_OF_SAMPLE_ROW-1, 1, -1):
        if (Avg_of_each_Sample_row[i]>170 or Avg_of_each_Sample_row[i]<150):
            temp_straight_flag=0
    #done detect straght 2

    if ((not CROSS_FLAG) and(STRAIGHT_FLAG == 1 or temp_straight_flag == 1)):
        Detect_FLAG=STRAIGHT

    #detect curve and extreme curve
    if ((not CROSS_FLAG) and((Avg_of_each_Sample_row[1]!=160 and abs(Avg_of_each_Sample_row[1]-Avg_of_each_Sample_row[2])>37)
    or Avg_of_each_Sample_row[1]>200 or Avg_of_each_Sample_row[1]<120
    or (Avg_of_each_Sample_row[1]==160 and (Avg_of_each_Sample_row[2]>180 or Avg_of_each_Sample_row[2]<140)))):
        Detect_FLAG=CURVE
        if (Avg_of_each_Sample_row[1]>230 or Avg_of_each_Sample_row[1]<90):
            Detect_FLAG=EXTREME_CURVE
    

    #detect special curve
    if (Detect_FLAG==CURVE or Detect_FLAG==EXTREME_CURVE):
        line_row = image[REFERENCE_POINT_OF_CURVE, :]
        min_x = 0
        max_x = 320

        for x, y in enumerate(line_row):
            if y > 210:
                min_x = x
                break

        for x, y in enumerate(reversed(line_row)):
            if y > 210:
                max_x = 320-x
                break
        reference_point_x = int((max_x+min_x)/2)
        for i in range (1,3):
            temp_flag = 0
            min_xi, max_xi = Min_Max_of_Sample_Row[i]
            if (min_xi>0 or max_xi<320):
                if (reference_point_x <160 and Avg_of_each_Sample_row[i]>160):
                    Detect_FLAG = SPECIAL_CURVE
                    for j in range (3, 0, -1):
                        if (Avg_of_each_Sample_row_detail[j]>160):
                            REFERENCE_POINT_OF_SPECIAL_CURVE = 85+j*10
                            temp_flag = 1
                            break
                if (reference_point_x >160 and Avg_of_each_Sample_row[i]<160):
                    Detect_FLAG = SPECIAL_CURVE
                    for j in range (3, 0, -1):
                        if (Avg_of_each_Sample_row_detail[j]<160):
                            REFERENCE_POINT_OF_SPECIAL_CURVE = 85+j*10
                            temp_flag = 1
                            break 
            if(temp_flag==1):
                break
    #stablize when in straight
    if(Detect_FLAG==STRAIGHT):
        # x1, y1 lower left
        # x2, y2 upper left
        # x3, y3 lower right
        # x4, y4 upper right
        x1, x3 = Min_Max_of_Sample_Row[4]
        y1 = 85+4*10
        y3 = 85+4*10
        x2, x4 = Min_Max_of_Sample_Row[1]
        y2 = 85+1*10
        y4 = 85+1*10
        x, y = IntersecFromFourPoint(x1, y1, x2, y2, x3, y3, x4, y4)
        image = cv2.circle(image, (x, y), 2, 255)
        if(x>110 and x < 220):
            line_row = image[REFERENCE_POINT_OF_STRAIGHT, :]
            min_x = 0
            max_x = 0

            for x1, y1 in enumerate(line_row):
                if y1 > 210:
                    min_x = x1
                    break

            for x1, y1 in enumerate(reversed(line_row)):
                if y1 > 210:
                    max_x = 320-x1
                    break
            avg_x = int((min_x+max_x)/2)
            ref_x = int((avg_x - x)/3+x)
            ref_y = int((REFERENCE_POINT_OF_STRAIGHT-y)/3+y)
            image = cv2.circle(image, (ref_x, ref_y), 1, 40, 2)
            x0, y0 = int(w/2), h

            if (y0 == ref_y):
                value = 0
            else:
                value = (ref_x-x0)/(y0-ref_y)

            angle = math.degrees(math.atan(value))
            return angle/8, DEFAULT_SPEED

    if(Detect_FLAG==STRAIGHT):
        CHECKPOINT = REFERENCE_POINT_OF_STRAIGHT
        speed = DEFAULT_SPEED
    if(Detect_FLAG==CURVE):
        CHECKPOINT = REFERENCE_POINT_OF_CURVE
        speed = CURVE_SPEED
    if(Detect_FLAG==EXTREME_CURVE):
        CHECKPOINT = REFERENCE_POINT_OF_EXTREME_CURVE
        speed = CURVE_SPEED
    if(Detect_FLAG==SPECIAL_CURVE):
        CHECKPOINT = REFERENCE_POINT_OF_SPECIAL_CURVE
        speed = CURVE_SPEED

    line_row = image[CHECKPOINT, :]
    min_x = 0
    max_x = 0

    for x, y in enumerate(line_row):
        if y > 210:
            min_x = x
            break

    for x, y in enumerate(reversed(line_row)):
        if y > 210:
            max_x = 320-x
            break
    center_row = int((max_x+min_x)/2)
    image = cv2.circle(image, (center_row, CHECKPOINT), 1, 40, 2)
    
    x0, y0 = int(w/2), h
    x1, y1 = center_row, CHECKPOINT

    value = (x1-x0)/(y0-y1)

    angle = math.degrees(math.atan(value))

    if (current_speed > speed):
        speed = 0

    if(Detect_FLAG==CROSS_ROAD):
        return angle/8,speed
    if(Detect_FLAG==STRAIGHT):
        if (angle>56):
            angle = 56
        return angle/8, speed
    if(Detect_FLAG==CURVE):
        return angle/3.7, speed
    if(Detect_FLAG==EXTREME_CURVE):
        return angle/3, speed
    if(Detect_FLAG==SPECIAL_CURVE):
        return angle/9, speed

    return angle/4, speed


if __name__ == "__main__":
    try:
        count = 0
        print("LTK TOI CHOI BRO")
        while True:
            state = GetStatus()
            
            raw_image = GetRaw()
            
            results= YOLO_model.predict(raw_image, conf=0.7, verbose=False)
            class_label = "none"

            for r in results:
                annotator = Annotator(raw_image)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    b = b.cpu().numpy().astype(int)
                    top, left, bottom, right = b
                    cropped_box = raw_image[left:right, top:bottom]
                    cropped_box = cv2.resize(cropped_box, (50, 50))
                    cropped_box = np.expand_dims(cropped_box, axis=0)
                    cropped_box = cropped_box.astype('float32')/255.0

                    traffic_sign = session1.run([output_name1], {input_name1: cropped_box})[0]
                    if(max(traffic_sign[0])>0.9):
                        annotator.box_label(b, class_dict[np.argmax(traffic_sign[0])])  
                        class_label=class_dict[np.argmax(traffic_sign[0])]    
                             
                                
            res = annotator.result()
            cv2.imshow("traffic", res)
            #Lane Detection
            img = cv2.resize(raw_image, (320, 176))
            input_data = np.expand_dims(img, axis=0).astype(np.float32)
            output = session.run([output_name], {input_name: input_data})
            segmentation_result = output[0]
            segmentation_result = (segmentation_result - np.min(segmentation_result)) / (np.max(segmentation_result) - np.min(segmentation_result)) * 255
            segment_image = segmentation_result.astype(np.uint8)
            cv2.imshow("segment", segment_image[0])


            angle, speed = AngCal(image=segment_image[0], current_speed=state["Speed"], class_label=class_label)
            # if(abs(angle)<10):
            #     AVControl(speed=20, angle=a)
            # else:
            AVControl(speed=speed, angle=angle)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
