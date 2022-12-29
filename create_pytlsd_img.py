
from plot_demo import IMG_PATH, TRAIN_IMG_NAMES_PATH, PYTLSD_PATH
from random import shuffle
from pathlib import Path
from skimage.transform import pyramid_reduce
import numpy as np
import cv2
import pytlsd

NOTDEF = -1024.0
# Script to create PyTSLD images and save them to disk

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

def create_pytlsd_image(id):
    pytlsd_im = analyze_image_pytlsd(IMG_PATH + f'{id}.jpg')[0]
    # print("Saving image", id)
    Path(PYTLSD_PATH).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(PYTLSD_PATH + f'{id}.jpg', pytlsd_im)

if __name__ == '__main__':
    with open(TRAIN_IMG_NAMES_PATH) as file:
        # Not really efficent but does the job
        lines = [line.rstrip() for line in file]
        shuffle(lines)
        ids = [line.split('.')[0] for line in lines]
        for id in ids:
            create_pytlsd_image(id)