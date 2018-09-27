"""
do pnet train
"""
import tensorflow as tf
import numpy as np


def keras_demo():
    """basic way to use tensorflow.keras
    """

    """create network
    model = tf.keras.Sequential(
        tf.keras.Dense(32, input_shape=(784,)),
        tf.kears.Activation('relu'),
    )
    """

    # we can also create network with .add
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=100))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # do train
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=10, batch_size=32)


if __name__ == '__main__':
    """test for tensorflow.keras
    """
    print('tf version: %s, keras version: %s' % (tf.__version__, tf.keras.__version__))
    keras_demo()
