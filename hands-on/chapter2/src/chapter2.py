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

import hashlib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer


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
    # def test_set_check(identifier, test_ratio, hash):
    #     return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

    # def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    #     ids = data[id_column]
    #     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    #     return data.loc[~in_test_set], data.loc[in_test_set]

    # housing_data_with_id = housing_data.reset_index()
    # train_set, test_set = split_train_test(housing_data_with_id, 0.2)

    # housing_data_with_id["id"] = housing_data["longitude"] * 1000 + housing_data["latitude"]
    # train_set, test_set = split_train_test_by_id(housing_data_with_id, 0.2, "id")

    train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
    logging.info('%d train + %d test' % (len(train_set), len(test_set)))


    housing_data["income_cat"] = np.ceil(housing_data["median_income"] / 1.5)
    housing_data["income_cat"].where(housing_data["income_cat"] < 5, 5.0, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
        strat_train_set = housing_data.loc[train_index]
        strat_test_set = housing_data.loc[test_index]

    logging.info(housing_data["income_cat"].value_counts() / len(housing_data))

    for set in (strat_train_set, strat_test_set): 
        set.drop(["income_cat"], axis=1, inplace=True)

    # Discover and Visualize the Data to Gain Insights
    # pass

    # Prepare the Data for Machine Learning Algorithms
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    ## Data cleaning
    imputer = Imputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
