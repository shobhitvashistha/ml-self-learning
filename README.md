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
the Y distance and not directly minimizing X distances


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

sigmoid function is used to transform domain to (0, 1) `S(x) = 1 / (1 + exp(-x))`

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


### Perceptron

The most basic form of a feed forward network

- decision function that receives inputs to produce a binary output
- output of 1 triggers the activation function while 0 does not
- in case of additional layers, 1 output can be configured to pass the output to next layer
- example activation function could be `>=0`, i.e. 1 output for non-negative numbers and 0 for negative ones
- weakness: output is binary, so a small change in weights/biases can produce polarizing results, which makes it difficult to train a model that is accurate with new data
- Sigmoid neuron - Alternative to perceptron with continuous output between 0-1 achieved by utilizing the sigmoid function `1 / (1 + exp(-x))`
- Hyperbolic tangent - Another alternative, `tanh(x)` produces output between -1 and 1, so it also outputs negative numbers, unlike sigmoid function
- Multilayer Perceptron (MLP) - Aggregate of multiple layers of perceptrons into a unified prediction model
- high number of hyperparameters mean it takes longer to run/train than other shallow models (can generally still be faster than SVM)


### Deep learning

A neural network with deep number of layers (at least 5-10). Can be useful in interpreting high number of features and
breaking down complex patterns into simpler patterns.

Types:
- Multilayer Perceptron (MLP)
- Recurrent Neural Network (RNN)
- Recursive Neural Tensor Network (RNTN)
- Deep Belief Network (DBN)
- Convolutional Neural Network (CNN)
- etc.

Applications:
- Text Processing (RNN, RNTN, CNN)
  - Sentiment analysis
  - Topic segmentation
  - Named entity recognition
- Image Recognition (DBN, CNN)
- Object recognition (RNTN, CNN)
- Speech recognition (RNN)
- Time series analysis (RNN)
- Classification (DBN, CNN, MLP)

Cons:
- Huge quantities of data needed (labeled or otherwise)
- High amount of resources and time needed for training the model
- Higher (when compared to other techniques) amount of resources and time needed for running the model
- Black-box - conceals the model's decision structure


## Decision Trees

Decision trees start at a root node and is followed by splits that produce branches, branches then link to leaves
(nodes) that form decision points.

The aim is to keep the tree as small as possible. this is achieved by selecting a variable that optimally splits
data into homogenous groups, such that it minimizes the level of data entropy at the next branch.

Entropy measures the variance in data among different classes.

```
Entropy (or information value) = - Σ Pi * log2(Pi)

where,

Pi = probability that a randomly selected item belongs to the i'th class
```

Tips:
- susceptible to over-fitting (because of the use of greedy algorithm) esp. for datasets with high pattern variance

### Bagging

Growing multiple decision trees using randomized selection of input data and combining results by averaging the
output (for regression) or voting (for classification)

Bootstrap sampling - extracting random variation of data each round, in case of bagging, different variations of the
training data is run through each tree

### Random Forests

Random Forests - Similar to bagging, except we artificially limit the number of variables considered for each split

With bagging the trees often look similar because they use the same variable early in their decision structure in a
bit to reduce entropy

High number of trees smooth out the potential impact of outliers, with diminishing returns beyond an adequate amount

### Boosting

Developing strong models by combining multiple weak models, achieved by adding weights to the trees based on
misclassified cases in a previous tree

#### Gradient Boosting 

Rather than selecting combinations of variables at random, we select variables that improve prediction accuracy with
each new tree. i.e. the trees are grown sequentially

At each iteration weights are added to the training data based on results of the previous iteration, with higher weights
applied to incorrectly predicted data, this process is repeated until there is a low level of error. The final result is
then obtained from a weighted average of the total predictions derived from each decision tree.

Notes:
- While adding more trees to a random forest helps offset over-fitting, the same can cause over-fitting in case of boosting
- Can lead to mixed results in case of data with high number of outliers, random forests may be preferable to boosting
- Slow processing speed because trees are trained sequentially, while random forests can be trained in parallel
- A downside that applies to boosting/bagging/random forests is the loss of clarity

## Ensemble modeling

Combine multiple algorithms/models to build a unified prediction model

Sequential models - prediction error is reduced by adding weights to classifiers that previously misclassified data
e.g. Gradient boosting, AdaBoost

Parallel models - prediction error is reduced by averaging e.g. Bagging, Random forests

Homogeneous models - combination of similar kinds of models (like bagging)

Heterogeneous models - combination of different kinds of models

Techniques:
- Bagging - parallel model averaging using a homogenous ensemble
- Boosting - homogenous technique reducing error by addressing misclassified cases in previous iteration to produce a sequential model
- Bucket of models - heterogeneous technique that trains multiple models using the same training data and selects one that performs best on test data
- Stacking - (usually heterogeneous) runs multiple models simultaneously and combines results to produce final model while adding emphasis to well-performing models


## Building a model in Python

Steps:

- import dataset
- scrub dataset
- split data into training and test data
- select an algorithm and configure its hyperparameters
- evaluate the results

[Example](examples/theobald/basic.py)


## Model optimization

Grid search - training/evaluating model using all combinations a given selection of hyperparameter values to
determine optimum values for each of the parameters

- It could be helpful to run a relatively coarse grid search first (say using powers of 10), and then run a finer grid search around the best value identified
- Randomized search can also be utilized to hone in on the optimal hyperparameters



