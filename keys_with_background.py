#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import sys
import argparse


# =============================================================================

def find_bounding_box(alpha_channel):
    """Find the bounding box of an image, based on its alpha channel

....The idea is to project the alpha value on the X and Y axis to get their histogram,
....then to define the box where 99,9% of the image opacity is.

....alpha_channel........Image alpha channel
....return ................(x, y, width, height) of the bounding box"""

# =============================================================================

    # Project image

    prj_h = np.sum(alpha_channel, axis=0)
    prj_v = np.sum(alpha_channel, axis=1)
    prj_tot = np.sum(prj_h)

    # This function finds the interval where the histogram is at (eps:1-eps) %

    def find_limits(projection, epsilon):

        integration = 0
        start = 0
        for i in range(len(projection)):
            integration += projection[i]
            if not start and integration > epsilon:
                start = i
            if integration > 1 - epsilon:
                end = i
                break

        return (start, end)

    # Find bounding box limits along X and Y

    (sx, ex) = find_limits(prj_h / prj_tot, 0.001)
    (sy, ey) = find_limits(prj_v / prj_tot, 0.001)

    return (sx, sy, ex - sx, ey - sy)


# =============================================================================

def load_key(
    path,
    size,
    rotation=0,
    flip=False,
    ):
    """Load a key image, with alpha channel included, and resize it, rotate it,
....and ajust its bounding box tightly

....path............Path to the image
....size............Highest dimension (width or height) of the final image [px]
....rotation........Rotation to apply to the image [deg]
....flip............True to flip the image along the horizontal axis

....return............Key image
...."""

# =============================================================================

    # Open image

    key = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    key = cv2.cvtColor(key, cv2.COLOR_BGRA2RGBA)
    if flip:
        key = cv2.flip(key, 0)
    (height, width) = key.shape[:2]

    # Increase canvas size to ensure to make any rotation without losing pixels

    dim = int(max(height, width) * 2 ** 0.5)
    new_canvas = np.zeros((dim, dim, 4), dtype=np.uint8)

    offx = (dim - width) // 2
    offy = (dim - height) // 2
    new_canvas[offy:offy + height, offx:offx + width, :] = key
    key = new_canvas

    # Apply the rotation

    rot_mtx = cv2.getRotationMatrix2D((dim // 2, dim // 2), 45, 1)
    key = cv2.warpAffine(key, rot_mtx, (dim, dim))

    # Find bounding box and remove what is outside

    alpha_channel = key[:, :, 3]
    (x, y, w, h) = find_bounding_box(alpha_channel)
    key = key[y:y + h, x:x + w, :]
    (height, width) = (h, w)

    # Resize image so that its highest dimension is 'size'

    f_width = width / size
    f_height = height / size
    f = max(f_width, f_height)
    key = cv2.resize(key, None, fx=1 / f, fy=1 / f,
                     interpolation=cv2.INTER_AREA)
    (height, width) = key.shape[:2]

    return key


# =============================================================================

def load_background(
    path,
    target_width,
    target_height,
    flip=False,
    ):
    """Load a background image, while ensuring its size fits the requested one
....If needed, image is cropped to preserve the aspect ratio

....path............................Path to the image
....target_width, target_height........Desired dimensions
....return ............................Image"""

# =============================================================================

    background = cv2.imread(path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    if flip:
        background = cv2.flip(background, 1)

    # Find scaling factor, so that the image is just bigger than requested (one dimension fit, the other is bigger)

    (height, width) = background.shape[:2]
    f_width = width / target_width
    f_height = height / target_height
    f = min(f_width, f_height)

    # Resize

    background = cv2.resize(background, None, fx=1 / f, fy=1 / f,
                            interpolation=cv2.INTER_AREA)
    (height, width) = background.shape[:2]

    # Then crop what is outside the requested size, with a random offset

    (height, width) = background.shape[:2]
    if height > target_height:
        offset = int(np.random.uniform(0, height - target_height))

        # offset = (height-target_height)//2

        background = background[offset:offset + target_height, :, :]
    elif width > target_width:

        offset = int(np.random.uniform(0, width - target_width))

        # offset = (width-target_width)//2

        background = background[:, offset:offset + target_width, :]

    return background


# =============================================================================

def addkey_to_background(
    background,
    key,
    x,
    y,
    blurr=0,
    ):
    """Patch a background image by the addition of a foreground image (key).
....The two images are combined by mixing them according to the key alpha channel

....background............Background image to patch
....key....................Image of the key to add to the background
....x, y................Position in the background image where to add the key
....blurr................Quantity of blurring (0..1)
...."""

# =============================================================================

    key_alpha = key[:, :, 3]
    key_rgb = key[:, :, :3]
    (height, width) = key.shape[:2]

    # For each alpha value at x, y position, create a triplet of this same value

    alpha_factor = 0.9 * key_alpha[:, :, np.newaxis].astype(np.float32) \
        / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor,
                                  alpha_factor), axis=2)

    # Compute the patch to apply to the image (mix of background and foreground)

    key_rgb = key_rgb.astype(np.float32) * alpha_factor
    patch = background.astype(np.float32)[y:y + height, x:x + width] \
        * (1 - alpha_factor)
    patch += key_rgb

    # patch the image

    background[y:y + height, x:x + width] = patch.astype(np.uint8)

    # A bit of blurring

    kernel_size = int(round(3 * blurr)) * 2 + 1
    blurred = cv2.GaussianBlur(background, (kernel_size, kernel_size),
                               0)
    return blurred


# =============================================================================
# """Image generation script"""
# =============================================================================

FLAGS = None

if __name__ == '__main__':

    # class YOLO defines the default value, so suppress any default here

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--keys', type=str, required=True,
                        help='path to keys path')

    parser.add_argument('--background', type=str, required=True,
                        help='path to background path')

    parser.add_argument('--output', type=str,
                        default='./keys_and_background',
                        help='path to output, default keys_and_background '
                        )
    FLAGS = parser.parse_args()

    NUM_IMAGES = 1000
    KEY_SIZE_RANGE = (60, 250)
    BACK_SIZE = 800

    PATH_KEYS = FLAGS.keys
    PATH_BACKGROUND = FLAGS.background
    PATH_OUTPUT = FLAGS.output
    os.mkdir(PATH_OUTPUT)
	

    # Load paths to key

    key_paths = []
    for path in os.listdir(PATH_KEYS):
        key_paths.append(os.path.join(PATH_KEYS, path))

    # Load paths to backgrounds

    back_paths = []
    for path in os.listdir(PATH_BACKGROUND):
        back_paths.append(os.path.join(PATH_BACKGROUND, path))

    csv_lines = []
    num_images = NUM_IMAGES
    while num_images > 0:

        # Choose configuration at random

        back_path = np.random.choice(back_paths)
        key_path = np.random.choice(key_paths)
        key_size = np.random.uniform(*KEY_SIZE_RANGE)
        x = int(np.random.uniform(BACK_SIZE - key_size - 1))
        y = int(np.random.uniform(BACK_SIZE - key_size - 1))
        angle = int(np.random.uniform(0, 360))
        flip = np.random.choice((True, False))
        flip_bckd = np.random.choice((True, False))
        blurr = np.random.uniform()

        # Combine background and foreground

        print (back_path)
        b = load_background(back_path, BACK_SIZE, BACK_SIZE, flip_bckd)
        k = load_key(key_path, key_size, angle, flip)
        final = addkey_to_background(b, k, x, y, blurr)

        # Save image

        output_path = os.path.join(PATH_OUTPUT,
                                   'gen_{:04d}.jpg'.format(num_images))
        img = image.array_to_img(final)
        img.save(output_path)

        # Keep track of image bounding box

        (height, width) = k.shape[:2]
        csv_lines.append('{} {},{},{},{},0\n'.format(output_path, x,
                         y, x + width, y + height))

        # plt.imshow (final)
        # plt.show ()

        num_images -= 1
        if num_images % 100 == 0:
            print (num_images, ' left')

    with open(os.path.join(PATH_OUTPUT, 'annotations.csv'), 'w') as f:
        for l in csv_lines:
            f.write(l)
