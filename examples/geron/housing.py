import os
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from examples import DATA_ROOT


RANDOM_STATE = 42  # the answer to life universe and everything


def load_data():
    return pd.read_csv(os.path.join(DATA_ROOT, 'housing', 'housing.csv'))


def hist(data):
    data.hist(bins=50, figsize=(12, 8))
    plt.show()


def shuffle_split_data(data, test_ratio=0.2):
    """
    We can shuffle and split the data set that way but that will cause us to have a different split everytime program is run
    Fixing the seed would mitigate the issue, but it will still occur whenever new datapoints are added
    """
    return train_test_split(data, test_size=test_ratio, random_state=RANDOM_STATE)


def is_id_in_test_set(identifier, test_ratio=0.2):
    return crc32(np.int64(identifier) < test_ratio * 2**32)


def split_data_with_id_hash(data, id_column, test_ratio=0.2):
    """
    Use id column's hash to split the dataset, this should produce a stable split
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def split_data_with_index(data, test_ratio=0.2):
    """
    add index column to be able to split the data by hash of the index column
    this should give us a stable way to split the data as long as newer data is appended to the end of previous data
    """
    data_with_id = data.reset_index()
    return split_data_with_id_hash(data_with_id, 'index', test_ratio)


def split_data_with_lat_long(data, test_ratio=0.2):
    """
    If updates to the data are not guaranteed to only append to the previous data, the solution would be to use stable
    features to construct an id column (lat long in this case)
    """
    data['id'] = data['longitude'] * 1000 + data['latitude']
    return split_data_with_id_hash(data, 'id', test_ratio)


def split_data_stratified(data, column, test_ratio=0.2):
    """
    We need our test data to be representative of the full dataset
    So if there is a feature that is important in determination of the estimate, it can be used to split the data
    """
    return train_test_split(data, test_size=test_ratio, random_state=RANDOM_STATE, stratify=data[column])


def split_data_stratified_multiple(data, column, n_splits=10, test_ratio=0.2):
    spliter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=RANDOM_STATE)
    stratified_splits = []
    for train_index, test_index in spliter.split(data, data[column]):
        stratified_splits.append([
            data.iloc[train_index], data.iloc[test_index]
        ])
    return stratified_splits


def add_income_category_column(data):
    data['income_cat'] = pd.cut(data['median_income'], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])


def show_income_category_hist(data):
    data['income_cat'].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel('Income category')
    plt.ylabel('Number of districts')
    plt.show()


def vis_geographic_data(data):
    data.plot(kind='scatter', x='longitude', y='latitude', grid=True, alpha=0.2)
    plt.show()


def vis_geographic_data_with_prices(data):
    data.plot(
        kind='scatter', x='longitude', y='latitude', grid=True,
        s=data['population'] / 100, label='population',  # size of the circles denotes population
        c='median_house_value', cmap='jet', colorbar=True,  # color of the circles denotes prices
        legend=True, sharex=True, figsize=(10, 7)
    )
    plt.show()


def get_correlations(data):

    # cor_matrix = data.corr()
    # above does not work because of presence of categorical values in the dataset
    cor_matrix = data.apply(lambda x: pd.factorize(x)[0]).corr()  # this works TODO: find out why?
    return cor_matrix['median_house_value'].sort_values(ascending=False)


def main_stratified():
    data = load_data()

    # add income category column for the stratified split
    add_income_category_column(data)

    # visualize income category histogram
    # show_income_category_hist(data)

    train_set, test_set = split_data_stratified(data, 'income_cat', 0.2)

    # we can see the proportions of the test set like so
    # test_set['income_cat'].value_counts() / len(test_set)
    # and compare it to the full data like so
    # data['income_cat'].value_counts() / len(data)

    # might as well delete the income category column now that we do not need it anymore
    for set_ in (train_set, test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    # make a copy for visualizations
    vis_data = train_set.copy()
    # vis_geographic_data(vis_data)
    # vis_geographic_data_with_prices(vis_data)

    print(get_correlations(vis_data))

