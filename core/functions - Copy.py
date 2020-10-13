import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

####################################################################


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def collide_warning(x, y):
    if len(detections) > 0:
        danger = dict()
        objectID = 2
        for dections in detections:
            name_tag = str(dections[2].decode())
            if name_tag == 'car':
                x, y, w, h = dections[2][0],\
                             dections[2][1],\
                             dections[2][2],\
                             dections[2][3],
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                danger[object_ID] = (int(x), int(y), xmin, ymin, xmax, ymax)
                objectID == 1

def collision_warning(data, stay_safe = True):
    boxes, scores, classes, num_objects = data
    
    if stay_safe:

        simRandomOutput = True
        #seperate corordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        #get midpoint
        #xMidpoint = (xmin + xmax)/2
        #yMidpoint = (ymin + ymax)/2

        #another way to get midpoint ??
        mid_x = (boxes[0][i][3] + boxes [0][i][1])/2
        mid_y = (boxes[0][i][2] + boxes [0][i][0])/2
        if i, b in enumerate(boxes[0])
          if relative_distance = round( (1 -(boxes[0][i][3] - boxes[0][i][1]))**4, 1)  
             if RELATIVE_DISTANCE <= 0.5:
                if mid_x > 0.3 and mid_x <0.7:
                    cv2.putText(image_np, '{WARNING}'.format(relative_distance), (int(mid_x*800) - 50,int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                    return simRandomOutput



###########################################################################


# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue
        
# function to run general Tesseract OCR on any detections 
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None
