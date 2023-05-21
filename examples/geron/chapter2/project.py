import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from examples import DATA_ROOT


RANDOM_STATE = 42  # the answer to life universe and everything


def load_housing_data():
    return pd.read_csv(os.path.join(DATA_ROOT, '../../../data/housing', 'housing.csv'))


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.kmeans_ = None

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())


def make_preprocessing_pipeline():
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, np.exp, feature_names_out="one-to-one"),
        StandardScaler())

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=RANDOM_STATE)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler())
    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ], remainder=default_num_pipeline)  # one column remaining: housing_median_age

    return preprocessing


def stratified_train_test_split(data, test_ratio=0.2, random_state=RANDOM_STATE):
    # add income category column for the stratified split
    data['income_cat'] = pd.cut(data['median_income'], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
    # do a stratified split on income_cat column
    train_set, test_set = train_test_split(data, test_size=test_ratio, random_state=random_state,
                                           stratify=data['income_cat'])
    # might as well delete the income category column now that we do not need it anymore
    for set_ in (train_set, test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    return train_set, test_set


def fit_predict_evaluate(model_name, model, data, data_labels):
    print(f'Starting training: {model_name}')
    model.fit(data, data_labels)
    print(f'Finished training: {model_name}')
    data_predictions = model.predict(data)

    error = mean_squared_error(data_labels, data_predictions, squared=False)
    print(f'- Error: {round(error, 2):.2f}\n- Percentage Error: {round(error * 100.0 / data_labels.mean()):.2f}%')

    print(f'Cross Validation: {model_name}')
    errors = -cross_val_score(model, data, data_labels, scoring="neg_root_mean_squared_error", cv=10)
    print(pd.Series(errors).describe())

    return error, errors


def main(test_ratio=0.2, random_state=RANDOM_STATE):
    housing = load_housing_data()

    train_set, test_set = stratified_train_test_split(housing, test_ratio=test_ratio, random_state=random_state)

    # test_set is not touched during training, train_set is copied
    # overwriting data to reference a copy of train_set so that we do not end up using the entire data for training
    housing = train_set.drop('median_house_value', axis=1)  # drop creates a copy too
    housing_labels = train_set['median_house_value'].copy()

    preprocessing = make_preprocessing_pipeline()

    # lin_reg = make_pipeline(preprocessing, LinearRegression())
    # tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=random_state))
    # forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=random_state))

    # fit_predict_evaluate('Linear regression', lin_reg, housing, housing_labels)
    # fit_predict_evaluate('Decision Tree regression', tree_reg, housing, housing_labels)
    # fit_predict_evaluate('Random Forest regression', forest_reg, housing, housing_labels)

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=random_state)),
    ])
    param_grid = [
        {'preprocessing__geo__n_clusters': [5, 8, 10],
         'random_forest__max_features': [4, 6, 8]},
        {'preprocessing__geo__n_clusters': [10, 15],
         'random_forest__max_features': [6, 8, 10]},
    ]
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                               scoring='neg_root_mean_squared_error', n_jobs=4)
    print('starting grid search')
    grid_search.fit(housing, housing_labels)
    print('grid search complete')
    print(f'best params: {grid_search.best_params_}')

    cv_res = pd.DataFrame(grid_search.cv_results_)
    cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

    # extra code – these few lines of code just make the DataFrame look nicer
    cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                     "param_random_forest__max_features", "split0_test_score",
                     "split1_test_score", "split2_test_score", "mean_test_score"]]
    score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
    cv_res.columns = ["n_clusters", "max_features"] + score_cols
    cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

    print(cv_res.head())

    return grid_search.best_estimator_


