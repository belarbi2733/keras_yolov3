#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image


def detect_img(yolo, image_path, output_path=''):
    try:
        image = Image.open(image_path)
    except:
        print ('Open Error! Try again!')
    r_image = yolo.detect_image(image)
    r_image.save(output_path)
    yolo.close_session()


FLAGS = None

if __name__ == '__main__':

    # class YOLO defines the default value, so suppress any default here

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--model', type=str,default='weights_yolo_train/trained_weights_stage_1.h5',
                        help='path to model weight file, default ' + YOLO.get_defaults("model_path"))

    parser.add_argument('--anchors', type=str,default='model_data/yolo_anchors.txt',
                        help='path to anchor definitions, default '
                        + YOLO.get_defaults('anchors_path'))

    parser.add_argument('--classes', type=str,default='model_data/key_classes.txt',
                        help='path to class definitions, default '
                        + YOLO.get_defaults('classes_path'))

    parser.add_argument('--gpu_num', type=int,
                        help='Number of GPU to use, default '
                        + str(YOLO.get_defaults('gpu_num')))

    parser.add_argument('--image', default=False, action='store_true',
                        help='Image detection mode, will ignore all positional arguments'
                        )
    parser.add_argument('--video', default=False, action='store_true',
                        help='video detection mode, will ignore all positional arguments'
                        )
    parser.add_argument('--input', nargs='?', type=str,
                        help='Video or image input path')

    parser.add_argument('--output', nargs='?', type=str, default='',
                        help='[Optional] Image or Video output path')

    FLAGS = parser.parse_args()

    if FLAGS.image:
        print ('Image detection mode')
        if 'input' in FLAGS:
            detect_img(YOLO(FLAGS.classes,FLAGS.anchors,FLAGS.model), FLAGS.input, FLAGS.output)
        else:
            print ('Must specify at least image_input_path.  See usage with --help.')
    elif FLAGS.video:
        if 'input' in FLAGS:
            detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
        else:
            print ('Must specify at least video_input_path.  See usage with --help.')
