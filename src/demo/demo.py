# tensorflow demo

import tensorflow as tf
import pandas
import numpy


def tensorflow_demo():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))


def pandas_demo():
    print(pandas.__version__)
    city_name = pandas.Series(['Beijing', 'Shanghai', 'Shenzhen'])
    population = pandas.Series([1000, 800, 700])

    data_frame = pandas.DataFrame({'City Name': city_name, 'Population': population})

    california_housing_dataframe = pandas.read_csv(
        "/Users/hankun/github/ml-learning/california_housing_train.csv", sep=",")
    print(california_housing_dataframe.describe())
    print(california_housing_dataframe.head())
