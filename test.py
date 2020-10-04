import os
import colorsys

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import cv2

from decode_np import Decode


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

if __name__ == '__main__':
    print('Please visit https://github.com/miemie2013/Keras-YOLOv4 for more complete model!')

    model_path = 'yolo4_weight.h5'
    model_path = 'logs/000/'+'ep018-loss25.846.h5' # voc 2007 neck
    model_path = 'logs/000/'+'ep046-loss6.901.h5' # raccoon neck
    model_path = 'ep009-loss3.856.h5' # raccoon fine tuned
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'
    #classes_path = 'model_data/raccoon_classes.txt'
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

    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    while True:
        img = input('Input image filename:')
        try:
            image = cv2.imread(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            image, boxes, scores, classes = _decode.detect_image(image, True)
            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    yolo4_model.close_session()
