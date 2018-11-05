"""
do pnet train
"""
import tensorflow as tf


def pnet_train():
    print ('do pnet train')
    input = tf.keras.layers.Input(shape=[12, 12, 3])
    x = tf.keras.layers.Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.lyaers.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = tf.keras.lyaers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

    classifier = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax', name='classifier1')(x)
    classifier = tf.keras.layers.Reshape((2,))(classifier)
    bbox_regress = tf.keras.layers.Conv2D(4, (1, 1), name='bbox1')(x)
    bbox_regress = tf.keras.layers.Reshape((4,))(bbox_regress)


if __name__ == '__main__':
    """test for tensorflow.keras
    """
    print('tf version: %s, keras version: %s' % (tf.__version__, tf.keras.__version__))
    pnet_train()
