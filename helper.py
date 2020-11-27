import os
import pandas as pd
import cv2 as cv


def load_scenes_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_images_from_folder(folder):
    images = []
    df = pd.DataFrame(columns=['label', 'scene_id', 'longitude', 'latitude'])
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            df = df.append({'label': filename[0], 'scene_id': filename.split('__')[1],
                            'longitude': filename.split('__')[2].split('_')[0],
                            'latitude': filename.split('__')[2].split('_')[1]}, ignore_index=True)
    return images, df


def to_num(string):
    return int(string)


def normalize(data):
    return [value / 255 for value in data]


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# Draw ROIs
def draw_rois(image, WinW, WinH, locations, color=(0, 0, 255)):
    img_holder = image.copy()
    for pt in zip(*locations):
        cv.rectangle(img_holder, pt, (pt[0] + WinW, pt[1] + WinH), color, 4)
    return img_holder
