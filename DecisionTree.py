import csv
import pydot
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.externals.six import StringIO  

def load_input_data():
    """ Load csv file containing input data and return as a numpy array
    """
    
    input_data = []
    with open('/Users/samimac2/Downloads/AcademicFeatureData.csv', newline = '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader: # Reading each row
            data_point = []
            for column in row: # Reading each column of the row
                data_point.append(float(column))
            input_data.append(data_point)
    input_data = np.array(input_data)
    
    return input_data


def load_target_data():
    """ Load csv file containing target data and return as a numpy array
    """
    
    target_data = []
    with open('/Users/samimac2/Downloads/Survey_Results.csv', newline = '') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            target_data.append([float(row[0])]) 
    target_data = np.array(target_data)
    
    return target_data

def scorer(estimator, X, target_data):
    """ Use squared sums as function for calculating score
    """
    
    estimator.fit(X, target_data)
    predictions = estimator.predict(X)
    score = ((predictions - y) ** 2).sum()
    
    return score

if __name__ == "__main__":
    input_data = load_input_data()
    target_data = load_target_data()

    print("Number of data points: ", len(target_data))

    # Normalize and standardize input data before training 
    normalized_X = preprocessing.normalize(input_data)
    standardized_X = preprocessing.scale(input_data)

    decisionTree = DecisionTreeClassifier(splitter='best', max_depth=2,random_state=3)

    # Cross validation using k-folds and leave one out
    cv = KFold(len(target_data), n_folds=3, shuffle=False)
    cv2 = LeaveOneOut(len(target_data))

    # Calculating scores for the model
    scores = cross_val_score(decisionTree, input_data, target_data, cv=cv, scoring=scorer)
    print("SCORES: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Value of the output when it was in the test set
    estimated_results = cross_val_predict(decisionTree, input_data, target_data, cv=cv)
    print("PREDICTED VALUES:", estimated_results)

    # Train the model
    model = DecisionTreeClassifier(splitter='best', max_depth=2,random_state=3)
    model.fit(input_data, target_data)
    expected = target_data
    predicted = model.predict(input_data)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    # Display decision tree as a graph
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(input_data, target_data)

    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph.write_svg("iris.svg")




