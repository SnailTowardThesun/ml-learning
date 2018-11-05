"""

"""

import tensorflow as tf
import numpy as np


def nn1_conv_net():
    print ('nn1 deep convolutional network')
    """build the nn1 convolutional network for face net, which also known as Zeiler&Fergus
    """
    model = tf.keras.Sequential()

    # conv1
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(7, 7), input_shape=(220, 220, 3), data_format='channels_last'))
    # pool1
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
    # rnorma2

    return model


def calibrate_box(bbox, reg):
    """
        calibrate bboxes
    Parameters:
    ----------
        bbox: numpy array, shape n x 5
            input bboxes
        reg:  numpy array, shape n x 4
            bboxes adjustment
    Returns:
    -------
        bboxes after refinement
    """
    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


if __name__ == '__main__':
    print ('do Face-Net training!')
    box = np.asarray([[373, 930, 413, 1330, 0.9998]])
    reg = np.asarray([-0.0956, -0.11, -0.122, -0.18])
    print(calibrate_box(box, reg))
