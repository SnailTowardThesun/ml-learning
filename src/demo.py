# tensorflow demo

import tensorflow as tf


def tensorflow_demo():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
