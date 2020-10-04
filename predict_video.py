#! /usr/bin/env python

import os
import argparse
import json
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tqdm import tqdm
import numpy as np

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import cv2

from decode_np import Decode

#from yolo4 import YOLO
#from .yolo4 import YOLO
from yolo import YOLO, detect_video

from matplotlib import pyplot as plt

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def _main_(args):
    input_path   = args.input
    output_path  = args.output

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45


    #model_path = 'ep073-loss11.905.h5'
    model_path = 'yolo4_weight.h5'
    model_path = 'logs/000/'+'ep018-loss25.846.h5'
    model_path = 'logs/000/'+'ep046-loss6.901.h5'
    model_path = 'logs/000/'+ 'ep019-loss3.684.h5'
    model_path = 'logs/000/'+'ep013-loss1.838.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    #classes_path = 'model_data/voc_classes.txt'
    classes_path = 'model_data/raccoon_classes.txt'
    #classes_path = 'model_data/coco_classes.txt'

    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)

    model_image_size = (608, 608)

    # 分数阈值和nms_iou阈值
    conf_thresh = 0.2
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)
    #print(yolo4_model.summary())
    print('num_anchors =', num_anchors)
    print('num_classes =', num_classes)


    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    print(model_path)
    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    #yolo = YOLO(model_path=model_path, anchors_path=anchors_path, classes_path=classes_path)

    ###############################
    #   Predict bounding boxes 
    ###############################
    if input_path[-4:] == '.mp4' or input_path[-5:] == '.webm': # do detection on a video  
        video_out = output_path + input_path.split('/')[-1]
        print('###')
        video_reader = cv2.VideoCapture(input_path)
        print('###')

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('input_path = ', input_path)
        print('nb_frames = ', nb_frames)
        print('frame_h = ', frame_h)
        print('frame_w = ', frame_w)

        #fourcc = cv2.VideoWriter_fourcc(*'mpeg')
        video_writer = cv2.VideoWriter(video_out,
                               #cv2.VideoWriter_fourcc(*'MPEG'),
                               cv2.VideoWriter_fourcc(*'mpeg'),
                               50.0,
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #if i == 100:
            #cv2.imshow('video with bboxes', image)
                #plt.imshow(image)
                #plt.show()
            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]
                for i in range(len(images)):
                #for i in range(1):
                    #cv2.imshow('video with bboxes', images[i])
                    #pass
                    # draw bounding boxes on the image using labels
                    #draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)
                    #plt.imshow(images[i])
                    #plt.show()
                    #cv2.imshow('video with bboxes', images[i])
                    images[i],_,_,_ = _decode.detect_image(images[i], True)

                    # show the video with detection bounding boxes          
                    #if show_window: cv2.imshow('video with bboxes', images[i])

                    # write result to the output video
                    video_writer.write(images[i])
                images = []
        video_reader.release()
        video_writer.release()
    else: # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
     
            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))         

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
