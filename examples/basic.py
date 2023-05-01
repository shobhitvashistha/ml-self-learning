import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cleaned_data():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'melbourne-housing-market', 'Melbourne_housing_FULL.csv'))

    # drop un-needed columns
    to_del = ['Address', 'Method', 'SellerG', 'Date', 'Postcode', 'Lattitude', 'Longtitude', 'Regionname',
              'Propertycount']
    for td in to_del:
        del df[td]

    # drop missing values
    df.dropna(axis=0, how='any', subset=None, inplace=True)

    # one-hot encode discrete columns
    df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])

    # get dependent and independent variables
    X = df.drop('Price', axis=1)
    y = df['Price']

    # split the dataset
    return train_test_split(X, y, test_size=0.3, shuffle=True)


def main():
    # get split up cleaned dataset
    X_train, X_test, y_train, y_test = cleaned_data()

    # select a model
    model = ensemble.GradientBoostingRegressor(
        n_estimators=250,  # number of decision trees
        learning_rate=0.1,  # rate at which additional decision trees influence the overall prediction
        max_depth=5,  # max depth of the decision trees
        min_samples_split=4,  # minimum number of samples required to execute a binary split
        min_samples_leaf=6,  # minimum number of samples that must appear in each child node before a new branch can be implemented
        max_features=0.6,  # total number of features used in determining the best split
        loss='huber'  # loss function
    )

    # loss function options
    # ls - least square regression - DOES NOT EXIST!!
    # lad - least absolute deviations - DOES NOT EXIST!!
    # quantile - quantile regression
    # huber - combination of ls and lad

    # train the model
    model.fit(X_train, y_train)

    # evaluate the results
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    print('Training Set Mean Absolute Error: %.2f' % mae_train)

    mae_test = mean_absolute_error(y_test, model.predict(X_test))
    print('Test Set Mean Absolute Error: %.2f' % mae_test)


def grid_search():
    # get split up cleaned dataset
    X_train, X_test, y_train, y_test = cleaned_data()

    # select a model
    model = ensemble.GradientBoostingRegressor()

    # define a selection of hyperparameters
    hyperparameters = {
        'n_estimators': [200, 300],
        'max_depth': [4, 6],
        'min_samples_split': [3, 4],
        'min_samples_leaf': [5, 6],
        'learning_rate': [0.01, 0.02],
        'max_features': [0.8, 0.9],
        'loss': ['quantile', 'huber', 'squared_error', 'absolute_error'],
    }

    # define grid search, n_jobs set to number of CPUs to parallelize the search
    grid = GridSearchCV(model, hyperparameters, n_jobs=6)

    # run grid search
    grid.fit(X_train, y_train)

    # to get the optimal params
    best_params = grid.best_params_
    print('Optimum params: ')
    print(best_params)

    # evaluate the model on optimal params
    # TODO: lol, this fails hard...
    # Training Set Mean Absolute Error: 36419212090929579606031908847991120583355375677406661086891522757734530696790213119142379062963470336.00
    # Test Set Mean Absolute Error: 36280561602259258959758476202678770733224505398907002058035428684486290819352956445160739918000947200.00
    # XD lol
    mae_train = mean_absolute_error(y_train, grid.predict(X_train))
    print('Training Set Mean Absolute Error: %.2f' % mae_train)

    mae_test = mean_absolute_error(y_test, grid.predict(X_test))
    print('Test Set Mean Absolute Error: %.2f' % mae_test)