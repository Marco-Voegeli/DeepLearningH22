# -*- coding: utf-8 -*-
from random import shuffle
import cv2
import pickle
import matplotlib.pyplot as plt

IMG_PATH = 'wireframe_dataset/v1.1/train/'
POINT_LINES_PATH = 'wireframe_dataset/pointlines/'
LETR_PATH = 'wireframe_dataset/letr/'
TRAIN_IMG_NAMES_PATH = 'wireframe_dataset/v1.1/train.txt'
PYTLSD_PATH = 'wireframe_dataset/pytlsd/'

# Load wireframe images


def load_img_points(idx):
    filename = POINT_LINES_PATH + f'{idx}.pkl'
    infile = open(filename, 'rb')
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
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 255, 0), 2, cv2.LINE_8)

    img_id = f'{id}.jpg'
    pytlsd_im = plt.imread(PYTLSD_PATH + img_id)
    letr_im = plt.imread(LETR_PATH + img_id)

    # Plot images
    fig, (row1, row2) = plt.subplots(2, 2, figsize=(10, 10))
    ax1, ax2, ax3, ax4 = row1[0], row1[1], row2[0], row2[1]
    axes = [ax1, ax2, ax3, ax4]

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

    plt.savefig(f'demo/{id}.png')


if '__main__' == __name__:
    with open(TRAIN_IMG_NAMES_PATH) as file:
        # Not really efficent but does the job
        lines = [line.rstrip() for line in file]
        ids = [line.split('.')[0] for line in lines]
        shuffle(ids)

        counter = 0
        NB_OF_DEMO_IMAGES = 20
        for id in ids:
            if counter < NB_OF_DEMO_IMAGES:
                try:
                    print_original_and_image_with_line(id)
                    counter += 1
                except FileNotFoundError:
                    print(f'Could not load {id}')

    plt.show()
