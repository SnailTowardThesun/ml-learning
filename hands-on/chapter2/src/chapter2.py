"""
 * Copyright(c) 2018 by hanvskun@hotmail.com
 * All rights reserved.
 *
 *  Author
 *      - hankun <hanvskun@hotmail.com>
"""


import sys
import os
import tarfile

import logging
import urllib.request

import pandas as pd
import numpy as np


def fetch_housing_data(housing_path, housing_url):
    """
    Download housing data from server
    Args:
        housing_path, the path to store tgz file
        housing_url, the url of tgz file
    Return:
        None
    """

    tgz_path = os.path.join(housing_path, "housing.tgz")
    logging.info('do download housing data from %s to %s' % (housing_url, tgz_path))

    if os.path.exists(housing_path):
        return

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    urllib.request.urlretrieve(housing_url, tgz_path)

    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """
    load housing data from csv file.
    Args:
        housing_path, the folder containing csv file
    Return:
        object of pandas storing data from csv
    """
    logging.info('do load housing data')
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    """
    get train and test data from data set
    Args:
        data, the container of data
        test_ration, the ration of test data.
    Return:
        [[train data], [test data]]
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()])

    download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    housing_path = "datasets/housing"
    housing_url = download_root + housing_path + "/housing.tgz"

    # download data from server
    fetch_housing_data(housing_path, housing_url)

    # load csv data
    housing_data = load_housing_data(housing_path)
    housing_data.info()

    # get train and test data
    train_set, test_set = split_train_test(housing_data, 0.2)
    logging.info('%d train + %d test' % (len(train_set), len(test_set)))
