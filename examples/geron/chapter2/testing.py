import os
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer, StandardScaler

from examples import DATA_ROOT


RANDOM_STATE = 42  # the answer to life universe and everything


def load_data():
    return pd.read_csv(os.path.join(DATA_ROOT, '../../../data/housing', 'housing.csv'))


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


def show_scatter_matrix(data):
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(data[attributes], figsize=(12, 8))
    plt.show()


def show_scatter_plot(data, x, y):
    data.plot(kind='scatter', x=x, y=y, alpha=0.1, grid=True)
    plt.show()


def add_experimental_features(data):
    data['rooms_per_house'] = data['total_rooms'] / data['households']
    data['bedrooms_ratio'] = data['total_bedrooms'] / data['total_rooms']
    data['people_per_house'] = data['population'] / data['households']


def handle_missing_values(data):
    # we could drop the rows with missing values
    # data.dropna(subset=["total_bedrooms"], inplace=True)  # option 1

    # we could drop the feature all togather
    # data.drop("total_bedrooms", axis=1)  # option 2

    # or we could fill the missing values with the median of the feature
    # median = data["total_bedrooms"].median()  # option 3
    # data["total_bedrooms"].fillna(median, inplace=True)

    # to display the rows with null values
    null_rows_idx = data.isnull().any(axis=1)
    data.loc[null_rows_idx].head()

    # instantiate an imputer
    imputer = SimpleImputer(strategy="median")
    # include only numerical types
    data_num = data.select_dtypes(include=[np.number])
    # fit data to imputer, this calculates the median values for each feature
    imputer.fit(data_num)

    print(imputer.statistics_)

    # transform the data, i.e. fill missing values with the median value, this will return a numpy array
    X = imputer.transform(data_num)
    # feature names can be obtained by
    print(imputer.feature_names_in_)

    # we can construct the data frame, by doing this
    data_tr = pd.DataFrame(X, columns=data_num.columns, index=data_num.index)

    # for scikit-learn >= 1.2
    # from sklearn import set_config
    # set_config(transform_output="pandas")

    return data_tr, X, imputer


def handle_categorical_features(data):
    data_cat = data[["ocean_proximity"]]
    ordinal_encoder = OrdinalEncoder()
    data_cat_encoded = ordinal_encoder.fit_transform(data_cat)

    print(ordinal_encoder.categories_)

    cat_encoder = OneHotEncoder()
    data_cat_1hot = cat_encoder.fit_transform(data_cat)  # this is a sparse matrix

    # to get dense array
    data_cat_1hot = data_cat_1hot.toarray()
    # or
    # cat_encoder = OneHotEncoder(sparse=False)

    print(cat_encoder.feature_names_in_)
    print(cat_encoder.get_feature_names_out())


def drop_outliers(X, data):
    isolation_forest = IsolationForest(random_state=42)
    outlier_pred = isolation_forest.fit_predict(X)

    print(outlier_pred)

    return data.iloc[outlier_pred == 1]
    # data_labels = data_labels.iloc[outlier_pred == 1]




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

    # print(get_correlations(vis_data))

    # show_scatter_matrix(vis_data)
    # show_scatter_plot(data, x='median_income',  y='median_house_value')

    add_experimental_features(vis_data)
    print(get_correlations(vis_data))

    data_tr, X, imputer = handle_missing_values(vis_data)

    # data_tr = drop_outliers(X, data_tr)

