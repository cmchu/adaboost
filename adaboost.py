import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def adaboost(X, y, num_iter):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_weights = []
    weights = [1.0 for i in range(len(y))]
    for m in range(num_iter):
        stump = DecisionTreeClassifier(max_depth=1, random_state=1)
        stump = stump.fit(X, y, sample_weight=weights)
        predictions = stump.predict(X)

        err = sum([weights[i] for i in range(len(predictions)) if predictions[i] != y[i]]) / float(sum(weights))
        if err == 0:
            alpha = 1.0
        else:
            alpha = np.log((1 - err) / float(err))

        for index in range(len(weights)):
            if predictions[index] != y[index]:
                weights[index] = weights[index] * np.exp(alpha)

        trees += [stump]
        trees_weights += [alpha]

    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y

    assumes Y in {-1, 1}^n
    """
    weighted_predictions = np.array([range(len(X)) for i in range(len(trees))], dtype=np.float)
    for m in range(len(trees)):
        tree = trees[m]
        predictions = tree.predict(X)
        tree_weight = trees_weights[m]
        weighted_predictions[m] = [tree_weight*prediction for prediction in predictions]

    Yhat = [sum(weighted_predictions[:, col]) for col in range(weighted_predictions.shape[1])]

    Yhat = [-1.0 if y < 0 else 1.0 for y in Yhat]

    return Yhat


def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
    csv = np.genfromtxt(filename, delimiter=",")
    X = csv[:,0:csv.shape[1]-1]
    Y = csv[:,csv.shape[1]-1]
    return X, Y

def new_label(Y):
    """ Transforms a vector od 0s and 1s in -1s and 1s.
    """
    return [-1. if y == 0. else 1. for y in Y]

def old_label(Y):
    return [0. if y == -1. else 1. for y in Y]

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y)) 

def main():
    """
    This code is called from the command line via
    
    python adaboost.py --train [path to filename] --test [path to filename] --numTrees 
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    num_trees = int(args['numTrees'][0])

    train_X, train_Y = parse_spambase_data(train_file)
    test_X, test_Y = parse_spambase_data(test_file)
    train_Y = new_label(train_Y)
    test_Y = new_label(test_Y)

    trees, trees_weights = adaboost(train_X, train_Y, num_trees)
    train_yhat = adaboost_predict(train_X, trees, trees_weights)
    test_yhat = adaboost_predict(test_X, trees, trees_weights)
    train_yhat = old_label(train_yhat)
    test_yhat = old_label(test_yhat)
    train_Y = old_label(train_Y)
    test_Y = old_label(test_Y)

    # print accuracy
    acc_test = accuracy(np.array(test_Y), np.array(test_yhat))
    acc = accuracy(np.array(train_Y), np.array(train_yhat))
    print("Train Accuracy %.4f" % acc)
    print("Test Accuracy %.4f" % acc_test)

    # write predictions to a file
    test = np.hstack((test_X, np.reshape(np.array(test_yhat), (len(test_yhat),1))))
    np.savetxt("predictions.txt", test, fmt='%.6g', delimiter=",")


if __name__ == '__main__':
    main()

