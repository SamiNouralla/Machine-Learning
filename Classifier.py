from __future__ import print_function
import os
import subprocess
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def get_iris_data():
    """ Get the iris data, from local csv or pandas repo. """

    # Find csv file on disk
    if os.path.exists("NormalizedDataV1_Headers.csv"):
        df = pd.read_csv("NormalizedDataV1_Headers.csv")
        
    # Get csv file from pandas repository
    else:
        fn = "https://raw.githubusercontent.com/pydata/pandas/" + \
             "master/pandas/tests/data/iris.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download iris.csv")
        with open("iris.csv", 'w') as f:
            print("-- writing to local iris.csv file")
            df.to_csv(f)
            
    return df

def encode_target(df, target_column):
    """Add column to df with integers for the target.
    df -- pandas DataFrame.
    target_column -- column to map to int, producing new Target column.
    """
    
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    
    return (df_mod, targets)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.
    tree -- scikit-learn DecisionTree.
    feature_names -- list of feature names.
    """
    
    with open("tre.svg", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)
    command = ["svg", "-Tpng", "tre.svg", "-o", "tre.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

if __name__ == "__main__":

    # Get features and target list from iris data
    df = get_iris_data()
    df2, targets = encode_target(df, "Overall, how satisfied are you with your program?")
    features = list(df2.columns[1:23])

    # Fit the decision tree
    target_data = df2["Target"]
    input_data = df2[features]
    dt = DecisionTreeClassifier(criterion='entropy',min_samples_split=5, random_state=51)
    dt.fit(input_data, target_data)

    # Produce graphic visualization
    visualize_tree(dt, features) 
