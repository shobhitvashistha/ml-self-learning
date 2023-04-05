# Notes - Machine Learning for Absolute Beginners - Oliver Theobald

Computer Science > Data Science > Artificial Intelligence > Machine Learning (read `>` as `is superset of`)

Artificial Intelligence (AI), Machine Learning (ML), and Data Mining all come under Data Science.
There is some overlap between AI and ML, and also overlap between ML and Data Mining.

Data Mining - Using human intuition to gain insights into data

## Data Scrubbing

1. Feature selection - selecting important features and excluding unimportant ones
2. Row compression - combining rows for the dataset together (inconvenient with large data-sets)
3. One-hot encoding - expanding categorical features to multiple features with 0/1 values
4. Binning - assigning bins to continuous variables
5. Normalization - transforming features such that the values lie between a fixed range, typically (0, 1) or (-1, 1), not recommended for extreme values
6. Standardization - transforms features into a standard distribution with mean 0 and standard distribution 1, recommended for SVM, PCA or k-NN
7. Missing data - we can either discard the rows or fill them up using mode/median of the values

## Setting up data

Spilt validation - 70/30 or 80/20 training/testing split, important to randomize the order before the split

Model performance measuring techniques -
- For classification -
  - Area under the curve (AUC)
  - Receiver Operating Characteristic (ROC)
  - Confusion Matrix
  - Recall
  - Accuracy
- For numeric output -
  - Root mean squared error (RMSE)
  - Mean absolute error (MAE)

### Cross validation

Splitting data into various combinations of test/training data and testing each combination

Exhaustive cross validation - testing all possible combinations of training/test data split (un-common in practice)

k-fold validation - dividing data into k buckets and testing k times reserving one of the bucket for testing and other k-1 for training

### How much data?

Features should cover the entire list/range of values

General rule of thumb when the amount of features is low, samples should be 10 times the number of features, this rule breaks for higher
dimensions as there are too many combinations possible with continuous variables.

But there is also diminishing rate of return after an adequate volume of training data. 

For datasets less with than 10k samples, clustering and dimensionality reduction can be highly effective.

Regression analysis and classification algorithms more suitable for less than 100k samples.

## Linear Regression

Plot a hyperplane (a line in case of single variable linear Regression) through the data such that the sum of residual errors is minimized

residual error = predicted value - actual value

NOTE: This is not equivalent to geometric line fitting in XY plane as you originally thought, as we are only minimizing
the Y distance and not directly minimizing X distances #TODO


```
Formula: (single variable)

y = bx + a

a = y-intercept, where hyperplane crosses the y-axis, or value of y when x=0
b = slope  

a = [ Σy * Σx^2 - Σx * Σxy ] / [ n * Σx^2 - (Σx)^2 ]
b = [ n * Σxy - Σx * Σy ] / [ n * Σx^2 - (Σx)^2 ]

```

X variables can be discrete but Y variables are always continuous 

Multi-collinearity - when two or more variables are strongly linearly co-related. To avoid this we need to check each
combination of "independent variables" using a scatter-plot, pairplot or correlation score

Correlation score - 1 for perfect positive correlation, -1 for perfect negative correlation, 0 for no relationship

Objective here is for all independent variables to be correlated to the dependent variable but not to each other


## Logistic regression

when we want discrete predictions instead of continuous

sigmoid function is used to transform domain to (0, 1)

hyperplane in Logistic regression serves as a dividing boundary for classification rather than a trendline predicting values

Logistic regression has benefits in binary classification. for classification in more than 2 bins multinomial logistic
regression can be used, however we may be better off using SVM or decision trees

Tips:
- Dataset should be free of missing values
- No 2 independent variables should be strongly linearly correlated
- 30-50 data points for each output
- usually does not work well with large datasets or messy data (outliers, complex relationships, missing values)


## k-Nearest Neighbors (k-NN)

k-NN classifies data-points based on their position to nearby data-points

- useful to test many k combinations to find the best fit, and to avoid setting k to high or too low
- too low k will increase bias and lead to mis-classification
- too high k will be computationally expensive
- setting k to uneven number will eliminate the possibility of a statistical stalemate or an invalid result
- standardization is recommended due to k-NN being sensitive to the scale of variables
- works best with continuous variables, only include discrete variables if they are critical (as this will skew the results)

Cons:
- time for single prediction proportional to the number of total points
- since calculating and storing distances between data-points is expensive
- not recommended for large datasets or high dimensional data


## k-Means clustering


Algo:
- k existing data-points are nominated as cluster centroids
- data-points are assigned a cluster based on distance from the centroids
- while data-points are still changing clusters
  - centroid coordinates are updated based on the current data-points in the cluster (by taking mean of xyz coordinates)
  - data-points are assigned a cluster based on distance from the new centroids


Tips:
- can fasten up things by choosing centroids that are not nearby to each other
- not always able to reliably identify a final combinations of clusters
- standardization is recommended


### Setting k

k = 1, all dataset belongs to 1 cluster

k = n, all points form their own cluster

Tips for choosing k:
- Scree plot - plot of number of clusters to the Sum of Squared Error (SSE), the elbow where increasing the cluster size no longer changes the SSE significantly
- square root of n/2
- domain knowledge


## Bias and Variance

Bias - Gep between predicted value and actual value

Variance - How scattered the values are with respect to each other

Under-fitting -
- Model is too simple and inflexible, has not scratched the surface of the underlying patterns
- High prediction error in both training and test data
- Low variance + High bias
- Common Causes:
  - Model too simple
  - Insufficient data to be able to cover all possible combinations of features
  - Data not randomized before split

Over-fitting -
- Model is overly complex and flexible
- Low prediction error in training data, but high prediction error in test data
- High variance + Low bias
- Common Causes:
  - Model too complex
  - Data not randomized before split

Bias / Variance Trade-off -
- Ideally we want both low variance and low bias, but there is a trade-off
- As model complexity increases, it moves from `low-variance + high-bias` to `high-variance + low-bias`
- Both bias and variance contribute to the error, but we want to minimize the error
- So the optimum solution with the least error over test data, lies somewhere in between


## Support Vector Machines (SVM)

Plot a hyperplane dividing the data-points maximizing the margin, i.e. sum of distances of the plane from
the closest points on either side of the plane (unlike logistic regression where sum of distances from all points
is minimized). This offers additional support to cope with new data-points that may infringe on the decision boundary

Can be useful for entangling complex relationships and mitigating outliers and anomalies

Hyperparameter `C` -
- Boundary can be modified by changing the hyperparameter `C`
- Can regulate the extent to which misclassified cases are ignored
- Low `C` will result in a wide or soft margin, greater generalization and low penalty for misclassified cases
- High `C` will result in a narrow or hard margin, high penalty for misclassified cases, susceptible to over-fitting
- `C = 0` will remove penalty for misclassified cases
- Any value of `C < 1.0` adds regularization the model
- Trade-off between `wide-margin + more-mistakes` and `narrow-margin + fewer-mistakes`
- Optimal `C` can be chosen by trial and error, which can be automated using `grid search`


Tips:
- Powerful with higher-dimension data
- Variations with Kernel Trick - map from low dimension to high dimension space, allowing the possibility of classifying non-linear data using a linear decision boundary in higher dimensions
- Sensitive to feature scales, standardization recommended
- Processing time to train could be a drawback
- Not recommended for datasets with low feature-to-row ratio due to speed and performance constraints


## Artificial Neural Networks (ANN)

Analyzes data through a network of decision layers, where each `layer` consists of multiple interconnected `nodes` with
adjustable `weights` that fire or do not fire based on a certain criterion.

```
input     hidden        output
layer     layer(s)      layer

          O x O
O    x    O x O    x    O
O    x    O x O    x    O
          O x O
```


At each note of the ANN -

```
output = activation_function( weight * input + bias )
```

Cost is a measure of difference between model's predicted value and the actual value. The purpose of training is to
reduce the cost by adjusting weights until the model's predictions closely matches the actual output.

Feed-Forward: Evaluate the network from left to right (input layer to output layer) to arrive at an output

Back-propagation: Using cost to incrementally tweak the network's weights from right to left until the lowest possible
cost value is achieved.

Black-box: Tracing ANN's decision structure reveals little to no insight about how specific variables influence its
decision.

When to use an ANN?
- Problems with large number of input features and complex patterns
- Problems that are too difficult for computers to solve but are almost trivial to humans
