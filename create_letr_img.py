from pathlib import Path
from random import shuffle
from letr_script import analyse_image_letr
from plot_demo import IMG_PATH, LETR_PATH, TRAIN_IMG_NAMES_PATH, create_letr_image
import cv2

def create_letr_image(id):
    letr_im = analyse_image_letr(IMG_PATH, f'{id}.jpg')
    # print("Saving image", id)
    Path(LETR_PATH).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(LETR_PATH + f'{id}.jpg', letr_im)
    
if __name__ == '__main__':
    with open(TRAIN_IMG_NAMES_PATH) as file:
        # Not really efficent but does the job
        lines = [line.rstrip() for line in file]
        shuffle(lines)
        ids = [line.split('.')[0] for line in lines]
        for id in ids:
            create_letr_image(id)