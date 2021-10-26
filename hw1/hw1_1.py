"""
===================== PLEASE WRITE HERE =====================
- EE3700 Homework1 coding part.
- A brief explanation of this script, e.g. the purpose of this script, what 
can this script achieve or solve, what algorithms are used in this script...

- YuanJunSun
- https://github.com/SYJINTW/EE3700_ML
===================== PLEASE WRITE HERE =====================
"""


# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data file and return two numpy arrays, including x: features and 
# y: labels
def load_data(path):
    print('Load data.')
    """
    - Use the function 'numpy.genfromtxt' to load the data.
    """
    data = np.genfromtxt(path, delimiter=',')
        
    # Print the number of samples
    n_samples = len(data)
    print('Number of samples:', n_samples)
    
    # Split the data into features and labels.
    # According to the dataset description, the first column is class 
    # identifier.
    x = data[:, 1:]
    y = data[:, 0].astype(int)
    
    return data, x, y 

# Get the number of samples in each class
def class_distribution(y):
    """
    - According to the dataset description, the given data consist of three 
    classes, namely 1, 2, and 3.
    - Please calculate the number of samples in each class. You may use the 
    function 'numpy.bincount'.
    """    
    # ===================== PLEASE WRITE HERE =====================
    
    bin_array = np.bincount(y)
    n_class1 = bin_array[1]
    n_class2 = bin_array[2]
    n_class3 = bin_array[3]
    
    # ===================== PLEASE WRITE HERE =====================
       
    print('Number of samples in class_1:', n_class1)
    print('Number of samples in class_2:', n_class2)
    print('Number of samples in class_3:', n_class3)
    

# Split the data into training set and testing set
def split_dataset(x, y, testset_portion):
    print('Split dataset.')
    """
    - In order to prevent a ML model from seeing answers before making 
    predictions, data is usually divived into a training set and a testing 
    set. A training set will be used to train an ML model (e.g. a classifier or 
    a regressor) and a testing set will be used to evaluate the performance of 
    an ML model.
    - Please split the data (both x and y) into a training set and a testing 
    set according to the 'testset_portion'. That is, the testing set will 
    account for 'testset_portion' of the overall data. You may use the function
    'sklearn.model_selection.train_test_split'.    
    """
    # ===================== PLEASE WRITE HERE =====================
    
    random_state_value = 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testset_portion, random_state=random_state_value)
    print(random_state_value)
    print(x_train)
    # print(f'test_size: {testset_portion}')
    # print(f'x: {np.size(x)} / {np.size(x_train)} / {np.size(x_test)} / {np.size(x_test)/np.size(x)}')
    # print(f'y: {np.size(y)} / {np.size(y_train)} / {np.size(y_test)} / {np.size(y_test)/np.size(y)}')
    
    # ===================== PLEASE WRITE HERE =====================
    
    return x_train, x_test, y_train, y_test
    

# Standardize the values of each feature dimension.
def feature_scaling(x_train, x_test):
    print('Feature scaling.')
    """
    - By observing the features 'x' in the 'Variable explorer', you will find
    that the values of some feature dimensions are 1, 2, or 1x, while some are 
    1xxx. It shows the values of each feature dimension are at different 
    levels. If these features are directly fed to an ML model, the result will 
    be dominate by the feature dimension with large values. As a result, the 
    process of feature scaling is necessary, which will standardize each 
    feature dimension with mean being 0 and stardard deviation being 1.
    - Please standardize both training set and testing set. You may use the 
    function 'sklearn.preprocessing.StandardScaler'.    
    """
    # ===================== PLEASE WRITE HERE =====================
    
    # To fixed the problem when some features are too small or too big
    scaler = StandardScaler().fit(x_train)
    x_train_nor = scaler.transform(x_train)
    scaler = StandardScaler().fit(x_test)
    x_test_nor = scaler.transform(x_test)

    # ===================== PLEASE WRITE HERE =====================    

    return x_train_nor, x_test_nor

# Train a Naive Bayes classifier on x_train and y_train
def train(x_train, y_train):
    print('Start training.')
    """
    - After the preprocessing, we can now train an ML model. We will train a 
    Naive Bayes classifier, which is based on the Baysian Decision Theory.
    - Since the input features are continuous values, we choose Gaussian Naive 
    Bayes classifier.
    - Please use the function 'sklearn.naive_bayes.GaussianNB' to train a 
    classifier.    
    """    
    # ===================== PLEASE WRITE HERE =====================
    
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    
    # ===================== PLEASE WRITE HERE =====================    
        
    return clf

# Use the trained classifier to test on x_test
def test(clf, x_test):
    print('Start testing.')
    """
    - Now we can use the trained classifier to predict the classes on x_test
    - Likewise, please the function 'sklearn.naive_bayes.GaussianNB'.
    """
    # ===================== PLEASE WRITE HERE =====================
    
    y_pred_label = clf.predict(x_test)
    y_pred = clf.predict_proba(x_test)
    # print(y_pred_label)
    # print(y_pred)
    
    # ===================== PLEASE WRITE HERE =====================
    
    return y_pred_label


# Main
if __name__=='__main__':
    # Some parameters
    path = 'wine.data'
    testset_portion = 0.2
    
    results = []

    for i in range(10):
        # Load data
        data, x, y = load_data(path)
        class_distribution(y)
        
        # Preprocessing
        x_train, x_test, y_train, y_test = split_dataset(x, y, testset_portion)
        x_train_nor, x_test_nor = feature_scaling(x_train, x_test)
        
        # Classification: train and test
        clf = train(x_train_nor, y_train)
        y_pred = test(clf, x_test_nor)
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print('\nAccuracy:', round(acc, 3))

        results.append([testset_portion ,acc])

    print(results)
    df = pd.DataFrame(results, columns=['testset_portion', 'accuracy'])
    ax = sns.barplot(x='testset_portion', y='accuracy',data=df)
    plt.show()
    