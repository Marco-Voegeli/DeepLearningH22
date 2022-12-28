
from plot_demo import IMG_PATH, PYTLSD_PATH, TRAIN_IMG_NAMES_PATH, analyze_image_pytlsd
from random import shuffle
from pathlib import Path
import cv2


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