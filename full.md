# Notes - Hands on Machine Learning with Scikit-Learn, Keras & TensorFlow - by Aurélien Géron

A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its
performance on T, as measured by P, improves with experience E.

T - the task
E - experience, examples, training data
P - performance measure, cost/loss function (how bad?), or utility/fitness function (how good?)

Analysing large amounts of data to discover hidden patterns is called data-mining.

Where to use ML?
- If existing solutions require a lot of fine-tuning or a long list of rules
- Complex problems for which using a traditional approach yields no good solution
- Fluctuating environments
- Getting insights about complex problems with large amounts of data

## Classification of ML systems
- How they are supervised (supervised, unsupervised, semi-supervised, self-supervised etc.)
- Whether they can learn incrementally on the fly (online vs batch learning)
- Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in training data and building a predictive model (instance-based vs. model-based learning)

### Supervision
- Supervised learning - when training dataset has labels (or targets)
- Unsupervised learning - when training dataset has no labels
- Semi-supervised learning - when training dataset has both labeled and unlabeled data 
- Self-supervised learning - generating fully labeled dataset from fully unlabeled one
- Reinforcement learning - *agent* performs an action in the environment according to a policy to receive a reward (or penalty), updates policy to maximize reward, iterates to arrive at optimal policy

### Batch vs Online learning

- Batch learning or Offline learning
  - The system is incapable of learning incrementally and must be trained using all available training data 
  - Model rot or Data drift - Performance decays slowly as world and data continues to evolve but the model remains unchanged
  - Requires training from scratch on old and new data if the model needs to be maintained to keep up with the changes in data

- Online learning or Incremental learning
  - The system is capable of learning incrementally
  - New data is either fed individually or in mini-batches
  - Out-of-core learning - Online learning algos can be used to train models on huge datasets that con not fit in one machine's memory
  - Learning rate - how fast should the model adapt to the changing data
  - If bad data is fed to the system the performance will decline

### Instance-based vs Model-based learning

- Instance-based learning - The system learns examples by heart, then generalizes to new cases by using a similarity measure to compare them to learned examples
- Model-based learning - The system builds a model from the examples and then uses the model to make predictions


## Main Challenges of ML

- Insufficient quantity of training data
- Non-representative training data - training data should be representative of the new cases
- Poor-Quality data - training data is full of errors. outliers and noise
- Irrelevant features - features that do not impact the result significantly
- Overfitting the training data - model too complex
- Underfitting the training data - model too simple

## Testing and validating

Split the data into training and test set, rule of thumb: 80:20 for smaller data sets, lower test set ratio for more quantities of data

Generalization error - error of the trained model on test set

### Hyperparameter tuning and model selection

- Even with training and test split it is possible to overfit the model to all the data during the tuning process
- When evaluating different models and selecting one that performs best on test set, it is possible to select a model that overfits all the data

Holdout Validation - hold out a part of training data as validation set (or development/dev set), use it to evaluate several
candidate models that are trained on the reduced training set and select the one that performs best on the validation set.
After this we train the best performing model on full training data, and evaluate this model on the test set to get an
estimation of the generalization error

- if validation set is too small, we may end up selecting a suboptimal model
- if validation set is too large, then the models are trained on a smaller training set, which might not be representative of the new cases

Cross-validation - Validate candidate models on several random combinations of training & validation set, select the model
that performs best on average in all the splits. This is time/resource intensive.

### Data mismatch

Training/Test data obtained may not representative of the actual production data, the quantity of production-like data that can be obtained can be quite low

To detect and fix this issue -
- We can split the data obtained for development of the model into `Train` and `Train-dev` set, train the model on just the Train set, and we split the prod-like data into `Dev` and `Test` set
- If it performs poorly on `Train-dev` set, then model must have overfit the `Train` set, simplify or regularize the model, get more training data and clean it up better
- If it performs well on `Train-dev` set, then we can evaluate it on the `Dev` set, if it performs poorly, this can be a case of data mismatch
- This problem can be addressed by pre-processing the data obtained for development to make it look more like the one that will be present in production
- Once we have a model that performs well on `Train-dev` as well as `Dev` set, we can finally evaluate it on the `Test` set to gauge its performance in production

No Free Lunch Theorem - If we make no assumptions about the data, there is no reason to prefer one over the other,
the only way to know which will perform better is to evaluate them all, since that is not possible in practice, we make
some reasonable assumptions about the data


## End-to-end ML project

Multiple regression - many features
Uni-variate regression - one output

### Performance measure

RMSE - root mean squared error (l2 norm)
MAE - mean absolute error (l1 norm)

- Higher order norms are more sensitive to outliers (large values), while lower order ones are more tolerant
- For a Gaussian distribution of values RMSE performs very well and is generally preferred

Data snooping bias - seemingly interesting pattern in the test data the leads you to select a particular kind of machine
learning model, leading to the generalization error being too optimistic

### Train Test split

- Random Sampling - We can shuffle and split the data set that way but that will cause us to have a different split everytime program is run
  - Fixing the seed would mitigate the issue, but it will still occur whenever new datapoints are added
- Stable Sampling - One solution is to use id column's hash to split the dataset
  - If there is no id column, add index column to be able to split the data by its hash
  - This should give us a stable way to split the data as long as newer data is appended to the end of previous data
  - If this is not the case then the solution would be to use stable features to construct an id column
- Stratified Sampling - We need our test data to be representative of the full dataset
  - so if there is a feature that is important in determination of the estimate, it can be used to split the data

### Explore and visualize data to gain insights

#### Looking for correlations

Correlation coefficient -
- used to determine weather or not 2 features are linearly correlated
- between -1 and 1
- closer to 1 implies strong positive correlation
- closer to -1 implies strong negative correlation
- 0 implies no **linear** correlation
- note that correlation coefficient can still be zero if there is a non-linear correlation

Scatter Matrix - A matrix of scatter plots among a set of features
- can reveal correlations between features
- can reveal weather or not certain features have skewed histograms, and hence need to be rescaled (diagonals are histograms, as scatter plot of feature with itself is always a 45 degree line)
- can reveal data quirks such as, caps or quantization in data

#### Experiment with attribute combinations

If certain combinations of features provide a better correlation with the intended result it may be useful to include these as features


### Clean the data

- Handle missing feature values
  - Most ML algorithms cannot work with missing features
  - We can either drop the feature with missing values, or the data points
  - Imputation: we can also fill in the missing values with mean/median etc.
  - KNNImputer and Iterative Imputer are more complex options available
- Handle Outliers
- Handle categorical features, Ordinal or One-hot encoding
  - Ordinal encoding has tendency to place unrelated categories near each-other
  - One hot avoids this issue, but if number of categories is large it can lead to a huge number of features

### Feature scaling and Transformation

- min-max scaling, or normalization
- standardization, much less effected by outliers
- the aim is to be able to transform feature is such a way that it's histogram is symmetrical around the mean
- taking the log of feature can help here
- bucketization can also be used to achieve this goal
- for multi-modal distributions (more than one peak)




## Classification

### Confusion matrix

```
[
  # predicted negative    # predicted positive
  
  [True Negatives,        False Positives]      # actual label negative 
  [False Negatives,       True Positives ]      # actual label positive
]
```

- model prediction on x-axis, negative to positive
- actual label on the y-axis, negative to positive

Precision - What proportion of all positive predictions are true positives? `TP / (TP + FP)`
- a model with no FPs but at least one TP has a perfect precision score
- when it's important for the model to have the least amount of FPs while FNs are tolerated
- e.g. if a post is safe for kids, there should be no FPs, FNs are ok as posters can appeal the decision for a manual review

Recall - What proportion of positively labeled data are the true positives? `TP / (TP + FN)`
- a model with no FNs but at least one TP has a perfect recall score
- when it's important for the model to have the least amount of FNs while FPs are tolerated
- e.g. cancer detection, there should be no FNs, FPs are ok as we perform detailed testing to confirm

F1 Score - Harmonic Mean of `Precision` and `Recall`

```math
F_1 = {2 \over \frac{1}{Precision} + \frac{1}{Recall}}
```

```
                     2
F1 =  _____________________________
       1 / Precision +  1 / Recall
```