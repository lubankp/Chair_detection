
# --------------------------------------------------------------------------
# Augmentation

import albumentations as alb
import os
import cv2
import json
import numpy as np

def augmentation_fun():

    augmentor = alb.Compose([alb.RandomCrop(width=1080, height=1080),
                              alb.RandomBrightnessContrast(p=0.2),
                              alb.RandomGamma(p=0.2),
                              alb.RGBShift(p=0.2)],
                              bbox_params=alb.BboxParams(format='albumentations',
                                                         label_fields=['class_labels']))

    # img = cv2.imread(os.path.join('train', 'images', '1685446615.2698824.jpg'))
    #
    # with open(os.path.join('train', 'labels', '1685446615.2698824.json'), 'r') as f:
    #     label = json.load(f)
    #
    # coords = [0, 0, 0.00001, 0.00001]
    # coords[0] = label['shapes'][0]['points'][0][0]
    # coords[1] = label['shapes'][0]['points'][0][1]
    # coords[2] = label['shapes'][0]['points'][1][0]
    # coords[3] = label['shapes'][0]['points'][1][1]
    # coords = list(np.divide(coords, [1920, 1080, 1920, 1080]))
    # augmented = augmentor(image=img, bboxes=[coords], class_labels=['chair'])
    #
    # cv2.rectangle(augmented['image'],
    #              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
    #              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)),
    #               (255, 0, 0), 2)
    #
    # cv2.imwrite(os.path.join('train', '1685446615.2698824.jpg'), augmented['image'])

    for partition in ['train', 'test', 'val']:
        for image in os.listdir(os.path.join(partition, 'images')):
            img = cv2.imread(os.path.join(partition, 'images', image))

            coords = [0, 0, 0.00001, 0.00001]
            label_path = os.path.join(partition, 'labels', image.split(".")[0] + "." + image.split(".")[1] + '.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)

                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                coords = list(np.divide(coords, [1920, 1080, 1920, 1080]))
            try:
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['chair'])
                    cv2.imwrite(os.path.join('aug_data', partition, 'images', image.split('.')[0] + '.' + image.split('.')[1] + str(x) + '.jpg'), augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0

                    with open(os.path.join('aug_data', partition, 'labels', image.split('.')[0] + '.' + image.split('.')[1] + str(x) + '.json'), 'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)
