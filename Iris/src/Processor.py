import os
import pandas as pd
import numpy as np
import random
import time


#this method is meant to process the iris.csv data & get it ready for machine learning
def process():
    #reading the data
    os.chdir("/Users/64000340/Desktop/iris-species")
    df = pd.read_csv("Iris.csv")
    k = 3 #number of classes
    #assigning discrete values to the type of plant
    transformed = []
    transformer = {"Iris-setosa": 0, "Iris-virginica": 1, "Iris-versicolor": 2}
    for index in list(df["Species"]):
        transformed.append(transformer.get(index))
    #appending the new discrete value column into the dataframe
    df['species_value'] = transformed
    df = df.drop('Species', axis = 1)
    #getting means & standard deviations ready for feature scaling
    tb = df.describe()
    means = []
    stds = []
    for col in tb.columns:
        means.append(round(tb[col][1], 2))
        stds.append(round(tb[col][2], 2))
    #used to indicate which values can be scaled.
    '''
    In order to be scaled, the feature must be a continuous or quantitative variable like Height &
    so forth. The indicator dictionary tells the function feature scale what each variable 
    is. C = Categorical, and Q = Quantitative
    '''
    indicator = {'Id': "C",
                 'SepalLengthCm' : "Q",
                 'SepalWidthCm': 'Q' ,
                 'PetalLengthCm' : "Q",
                 'PetalWidthCm' : "Q",
                 'species_value' : 'C'}

    df_scaled = featureScale(means, stds, df, indicator)
    answers = list(df['species_value'])
    df_scaled = df_scaled.drop('species_value', 1)
    X, Y = convert_to_np(df, answers, k)
    X_train, Y_train, X_CV, Y_CV, X_test, Y_test = give_train_cv_test(X, Y)
    os.chdir("/Users/64000340/PycharmProjects/Iris")
    #creating new files with the training, cv, and test data
    with open("Train/X_train.mat", 'w') as f:
        np.savetxt(f, X_train)
    with open("Train/Y_train.mat", 'w') as f:
        np.savetxt(f, Y_train)
    with open("CV/X_CV.mat", 'w') as f:
        np.savetxt(f,X_CV )
    with open("CV/Y_CV.mat", 'w') as f:
        np.savetxt(f, Y_CV)
    with open("Test/X_test.mat", 'w') as f:
        np.savetxt(f, X_test )
    with open("Test/Y_test.mat", 'w') as f:
        np.savetxt(f, Y_test)

'''
The Function below will alter the original dataframe
'''
#returns the dataframe after it has been scaled
def featureScale(means, stds, df, indicator):
    temp_df = df
    colIndex = 0
    #iterating over columns to check the type of variable the column is
    for col in df.columns:
        if(indicator.get(col) == 'C'):
            print(col, "is a categorical variable")
        else:
            for i in range(len(list(df[col]))):
                #subtracting by the mean & finding the z-score

                temp_df[col][i] -= means[colIndex]
                temp_df[col][i] /= stds[colIndex]
        colIndex+=1

    return df

#creates a one hot vector for answer labels & converts all data into numpy
def convert_to_np(df, answers, k):
    #getting the label into a one hot vector
    Y = np.zeros((1, k))
    for val in answers:
        list = [0] * 3
        list[val] = 1
        Y = np.vstack((np.array(list), Y))

    Y = Y[:np.shape(Y)[0] - 1]
    return df.values, Y

#returns
def give_train_cv_test(X, Y):
    X_train = np.zeros((1, np.shape(X)[1]))
    Y_train = np.zeros((1, np.shape(Y)[1]))
    X_CV = np.zeros((1, np.shape(X)[1]))
    Y_CV = np.zeros((1, np.shape(Y)[1]))


    total = np.shape(X)[0] #total amount to begin with
    while (np.shape(X)[0] > 0.2 * total):
        random_number = random.randrange(0, np.shape(X)[0])
        #appending the items
        X_train = np.vstack((X_train, X[random_number, :]))
        Y_train = np.vstack((Y_train, Y[random_number, :]))
        #removing the number from the list
        if(random_number == np.shape(X)[0] - 1):
            X = X[:random_number]
            Y = Y[:random_number]
        else:
            X = np.vstack((X[:random_number], X[random_number + 1:]))
            Y = np.vstack((Y[:random_number], Y[random_number + 1:]))

    total = np.shape(X)[0]
    while (np.shape(X)[0] > 0.5 * total):
        random_number = random.randrange(0, np.shape(X)[0])
        #appending items
        X_CV = np.vstack((X_CV, X[random_number, :]))
        Y_CV = np.vstack((Y_CV, Y[random_number, :]))
        #removing the numbers
        if(random_number == np.shape(X)[0] - 1):
            X = X[:random_number]
            Y = Y[:random_number]
        else:
            X = np.vstack((X[:random_number], X[random_number + 1:]))
            Y = np.vstack((Y[:random_number], Y[random_number + 1:]))

    X_test = X
    Y_test = Y


    return X_train[1:], Y_train[1:], X_CV[1:], Y_CV[1:], X_test, Y_test









start = time.time()
process()
print(time.time() - start)

