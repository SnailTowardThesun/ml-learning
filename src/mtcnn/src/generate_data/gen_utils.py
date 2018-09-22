"""
some utils function to handle wider data
"""
import os
import numpy as np


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square
    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox
    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


def parse_wider_recorder_file(recorder_file_path, images_folder):
    if len(recorder_file_path) == 0:
        raise Exception('the path of wider recorder file can not be empty')

    recorder_file = open(recorder_file_path, "r")

    dic = {}

    labels = []
    label_total_number = 0
    label_loaded_number = 0
    image_path = ''

    for line in recorder_file:
        line = line[:len(line) - 1]

        if len(image_path) == 0:
            image_path = line
            continue

        if label_total_number == 0:
            label_total_number = int(line)
            continue

        labels.append(line)
        label_loaded_number += 1

        if label_loaded_number == label_total_number:
            image_path = os.path.join(images_folder, image_path)
            dic[image_path] = labels

            labels = []
            image_path = ''
            label_loaded_number = label_total_number = 0

    recorder_file.close()

    return dic


if __name__ == '__main__':
    print ('generate utils test')
    recorder_path = '/Users/hankun/github/ml-learning/src/mtcnn/source/wider_face_split/' \
                    'wider_face_train_bbx_gt.txt'

    images_path = '/Users/hankun/github/ml-learning/src/mtcnn/source/WIDER_train/images'
    parse_wider_recorder_file(recorder_path, images_path)
