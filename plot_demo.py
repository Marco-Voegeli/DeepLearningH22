# -*- coding: utf-8 -*-
import cv2
import pytlsd
import pickle
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from letr_script import analyse_image_letr
from skimage.transform import pyramid_reduce

NOTDEF = -1024.0
IMG_PATH = 'wireframe_dataset/v1.1/train/'
POINT_LINES_PATH = 'wireframe_dataset/pointlines/'
TRAIN_IMG_NAMES_PATH = 'wireframe_dataset/v1.1/train.txt'

def get_thresholded_grad(resized_img):
    modgrad = np.full(resized_img.shape, NOTDEF, np.float64)
    anglegrad = np.full(resized_img.shape, NOTDEF, np.float64)

    # A B
    # C D
    A, B, C, D = resized_img[:-1, :-1], resized_img[:-1, 1:], resized_img[1:, :-1], resized_img[1:, 1:]
    gx = B + D - (A + C)  # horizontal difference
    gy = C + D - (A + B)  # vertical difference

    threshold = 5.2262518595055063
    modgrad[:-1, :-1] = 0.5 * np.sqrt(gx ** 2 + gy ** 2)
    anglegrad[:-1, :-1] = np.arctan2(gx, -gy)
    anglegrad[modgrad <= threshold] = NOTDEF
    return gx, gy, modgrad, anglegrad

def analyze_image_pytlsd(image_path='pyt_lsd/resources/ai_001_001.frame.0000.color.jpg'):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    flt_img = gray.astype(np.float64)

    scale_down = 0.8
    resized_img = pyramid_reduce(flt_img, 1 / scale_down, 0.6)

    # Get image gradients
    gx, gy, gradnorm, gradangle = get_thresholded_grad(resized_img)

    segments = pytlsd.lsd(resized_img, 1.0, gradnorm=gradnorm, gradangle=gradangle)
    segments /= scale_down

    # plt.title("Gradient norm")
    # plt.imshow(gradnorm[:-1, :-1])
    # plt.colorbar()
    # plt.figure()
    # gradangle[gradangle == NOTDEF] = -5
    # plt.title("Thresholded gradient angle")
    # plt.imshow(gradangle[:-1, :-1])
    # plt.colorbar()

    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for segment in segments:
        cv2.line(img_color, (int(segment[0]), int(segment[1])), (int(segment[2]), int(segment[3])), (0, 255, 0))

    return cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), segments, gradnorm, gradangle

# Load wireframe images

def load_img_points(idx):
    filename = POINT_LINES_PATH + f'{idx}.pkl'
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()

    im = new_dict['img']
    lines = new_dict['lines']
    points = new_dict['points']

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im, lines, points

# Show wireframe images without and with lines

def print_original_and_image_with_line(id):
    im, lines, points = load_img_points(id)
    imb_before = im.copy()
    for idx, (i, j) in enumerate(lines, start=0):
     x1, y1 = points[i]
     x2, y2 = points[j]
     cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_8)

    pytlsd_im = analyze_image_pytlsd(IMG_PATH + f'{id}.jpg')[0]

    letr_im = analyse_image_letr(IMG_PATH, f'{id}.jpg')

    # Plot images
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 20))

    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.imshow(imb_before)
    ax1.set_title('Original Image')

    ax2.imshow(im)
    ax2.set_title('with ground truth lines')

    ax3.imshow(pytlsd_im)
    ax3.set_title('with PyTSLD lines')

    ax4.imshow(letr_im)
    ax4.set_title('with LETR lines')
    
    plt.savefig(f'wireframe_dataset/demo_{id}.png')

with open(TRAIN_IMG_NAMES_PATH) as file:
    # Not really efficent but does the job
    lines = [line.rstrip() for line in file][:10]
    shuffle(lines)
    ids = [line.split('.')[0] for line in lines]
    for id in ids:
        print_original_and_image_with_line(id)
