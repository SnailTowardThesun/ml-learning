"""
generate data for PNET
"""
import numpy as np
import cv2
import os
import numpy.random as npr
from gen_utils import IoU, parse_wider_recorder_file


def generate_pnet_data():
    """generate data for PNET
    """
    print 'do generate data for PNET'

    pnet_source_root = './pnet_12'

    if not os.path.exists(pnet_source_root):
        os.mkdir(pnet_source_root)

    positive_folder_path = os.path.join(pnet_source_root, 'positive')
    negative_folder_path = os.path.join(pnet_source_root, 'negative')
    part_folder_path = os.path.join(pnet_source_root, 'part')

    if not os.path.exists(positive_folder_path):
        os.mkdir(positive_folder_path)
    if not os.path.exists(negative_folder_path):
        os.mkdir(negative_folder_path)
    if not os.path.exists(part_folder_path):
        os.mkdir(part_folder_path)

    positive_recorder_file_path = os.path.join(positive_folder_path, 'pos_recorder.txt')
    negative_recorder_file_path = os.path.join(negative_folder_path, 'neg_recorder.txt')
    part_recorder_file_path = os.path.join(part_folder_path, 'part_recorder.txt')

    positive_recorder_file = open(positive_recorder_file_path, 'w')
    negative_recorder_file = open(negative_recorder_file_path, 'w')
    part_recorder_file = open(part_recorder_file_path, 'w')

    recorder_path = '/Users/hankun/github/ml-learning/src/mtcnn/source/wider_face_split/' \
                    'wider_face_train_bbx_gt.txt'

    images_path = '/Users/hankun/github/ml-learning/src/mtcnn/source/WIDER_train/images'

    image_dic = parse_wider_recorder_file(recorder_path, images_path)

    num = len(image_dic)
    print "%d pics in total" % num
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # dont care
    idx = 0
    box_idx = 0

    loop = 0
    for im_path, annotations in image_dic.iteritems():
        # get only 200 pictures for demo because of poor cpu
        loop += 1
        if loop > 200:
            break

        annotation = []
        for item in annotations:
            tmp = item.split(' ')
            annotation += tmp[:4]
        bbox = map(float, annotation[:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

        img = cv2.imread(im_path)
        idx += 1
        if idx % 100 == 0:
            print idx, "images done"

        height, width, channel = img.shape

        neg_num = 0
        while neg_num < 15:
            size = npr.randint(48, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            iou = IoU(crop_box, boxes)

            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(negative_folder_path, "%s.jpg" % n_idx)
                negative_recorder_file.write("12/negative/%s" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            box[2] += box[0]
            box[3] += box[1]
            x1, y1, x2, y2 = box

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if min(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # generate positive examples and part faces
            for i in range(10):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[ny1: ny2, nx1: nx2, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(positive_folder_path, "%s.jpg" % p_idx)
                    positive_recorder_file.write("12/positive/%s" % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_folder_path, "%s.jpg" % d_idx)
                    part_recorder_file.write("12/part/%s" % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            print "%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx)

    positive_recorder_file.close()
    negative_recorder_file.close()
    part_recorder_file.close()


if __name__ == '__main__':
    """test for generate pnet data
    """
    print('generate data for PNET')
    generate_pnet_data()
