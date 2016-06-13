import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import csv

def load_input_data():
    """ Load csv file containing input data and return as a numpy array
    """
    
    input_data = []
    with open('/Users/samimac2/Desktop/PythonDataFiles/Top15NoheaderNoZero.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:  # Reading each row
            data_point = []
            for column in row:  # Reading each column of the row
                data_point.append(float(column))
            input_data.append(data_point)
    input_data = np.array(input_data)
    
    return input_data

def load_target_data():
    """ Load csv file containing target data and return as a numpy array
    """

    target_data = []
    with open('/Users/samimac2/Desktop/PythonProject/testResults.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            target_data.append([float(row[0])])
    target_data = np.array(target_data)

    return target_data

def scorer(estimator, X, y):
    """ Use squared sum as function for calculating score
    """
    
    estimator.fit(X, y)
    predictions = estimator.predict(X)
    score = ((predictions - y) ** 2).sum()  # it should return one value
    
    return score

if __name__ == "__main__":
    input_data = load_input_data()
    target_data = load_target_data()
    
    print("Number of data points: ", len(target_data))

    decisionTree = RandomForestClassifier(max_features="log2")

    # Cross validation using K-fold and leave one out
    cv = KFold(len(target_data), n_folds=2, shuffle=True)
    cv2 = LeaveOneOut(len(target_data))  # only needs the number of points

    # Calculating scores for the model
    scores = cross_val_score(decisionTree, input_data, target_data, cv=cv)
    print("SCORES: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Value of the output when it was in the test set
    estimated_results = cross_val_predict(decisionTree, input_data, target_data, cv=cv)
    print("PREDICTED VALUES:", estimated_results)

    # Train the model
    decisionTree.fit(input_data,target_data)
    predicted = decisionTree.predict(input_data)
    expected = target_data
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
