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
