import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

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
    # ls - least square regression
    # lad - least absolute deviations
    # quantile - quantile regression
    # huber - combination of ls and lad

    # train the model
    model.fit(X_train, y_train)

    # evaluate the results
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    print('Training Set Mean Absolute Error: %.2f' % mae_train)

    mae_test = mean_absolute_error(y_test, model.predict(X_test))
    print('Test Set Mean Absolute Error: %.2f' % mae_test)