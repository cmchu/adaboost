# Description

Implements AdaBoost from scratch, only using DecisionTreeClassifier from sci-kit learn to learn the base classifiers.

# Quick Review

Boosting methods combine the predictions of many weak/base classifiers in order to produce a strong classifier. In technical terms, AdaBoost is a forward stagewise additive model using the exponential loss function. In simpler terms, AdaBoost is a type of boosting method that sequentially trains weak classifiers each on different versions of the data. Weights are initialized to be equal for each training observation. Then, after training on the weak classifier, observations that were misclassified by that classifier have their weights increased, while observations that were correctly classified have their weights decreased. The next weak classifier is then trained on this modified data. This is repeated for a certain number of weak classifiers. Finally, the predictions of all of the weak classifiers are aggregated together through weighted voting in order to produce a single prediction for each sample. Inituitively, this algorithm works because observations that are continually difficult to classify correctly receive more and more weight, causing weak classifiers to concentrate more on them.

# Execution

This code requires the training and testing files to store the data as comma-delimited and have the response variables in the last column. Also, the response variable must only have 2 classes, which must consist of values of 0 and 1. Additionally, this code only uses stumps (decision trees with 1 level) for the base classifier.

The code can be run as follows: python adaboost.py --train path/to/train_set --test path/to/test_set

The code will print the train and test accuracy and will output a predictions.txt file, which will contain the predictions of the response variable. These predictions will have values of 0 and 1.
