# To do:
# 1) Tidy up code





##############################################################################
##############################################################################
##############################################################################
######                                                                  ######
######                        LOGISTIC REGRESSION                       ######
######                                                                  ######
##############################################################################
##############################################################################
##############################################################################



##############################################################################

##      This code implements an algorithm to solve classification           ##
##      problems using the logistic regression approach.                    ##

##############################################################################



"""

ALGORITHM STEPS

These are the steps that go into calculating the decision boundary:

    1) Parse the data.
    2) Normalize the data.
    3) Determine the coefficients of the decision boundary.
    4) Calculate various error metrics.
    5) Plot the data.


DATASET ASSUMPTIONS

These are the assumptions made about the input dataset:
    
    1) It is a .csv file.
    2) It is a 'tidy' dataset (look it up).
    3) You know exactly how you want to deal with anomalous entries.
    4) If it is a supervised problem, the last column is called 'Label'.
    5) No other columns are called 'Label' or 'Prediction'.


INSTRUCTIONS

This is what you need to do to run this program:

    1) Go to the section titled 'User Defined Parameters'.
    2) Read through each variable in this section.
    3) Replace the variables as appropriate.


"""



##########################################################################

##                           Import libraries                           ##

##########################################################################

import random
import numpy as np
import pandas as pd
import seaborn as sns
import collections as col
import matplotlib.pyplot as plt
import sklearn.datasets





class LogisticRegression:
    """ 

    Methods for calculating the logistic regression for classification problems.

        Attributes:     
            data: the dataset to be analyzed. 
            iterations: the number of iterations over which to train the model.

        Methods:    
            score(): determine fraction of correct predictions
            rmse(): determine root-mean square error
            r2(): determine Pearson correlation coefficient
            scikitlearn(): compare against scikit-learn implementation

        Usage:  
            >>> p = Performance(dataframe)
            
    """ 

    def __init__(self, data, labels, learning_rate=0.01, iterations=100000, threshold=0.5, verbose=True):
        self.data          = data
        self.labels        = labels
        self.learning_rate = learning_rate
        self.iterations    = iterations
        self.threshold     = threshold
        self.verbose       = verbose
    


    def sigmoid(self, z):
        """ Calculate the logistic function """
      
        return 1 / (1 + np.exp(-z))



    def loss(self, h):
        """ Calculate the cross-entropy loss """

        return (-self.labels * np.log(h) - (1 - self.labels) * np.log(1 - h)).mean()
    


    def fit(self):
        """ Calculate the weights using gradient descent """

        # Add a bias to the output 
        intercept = np.ones((self.data.shape[0], 1))
        self.data = np.concatenate((intercept, self.data), axis=1)
        
        # Initialize the weights
        self.theta = np.ones(self.data.shape[1])
        
        for i in range(self.iterations):
            z           = np.dot(self.data, self.theta)
            h           = self.sigmoid(z)
            gradient    = np.dot(self.data.T, (h - self.labels)) / self.labels.size
            self.theta -= self.learning_rate * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(self.data, self.theta)
                h = self.sigmoid(z)
                print(f'loss: {self.loss(h)} \t')
    


    def predict_prob(self):
        """ Determine the predicted class of an instance """
   
        return self.sigmoid(np.dot(self.data, self.theta))
    


    def predict(self):
        """ Determine to which class the output of an instance corresponds """

        return self.predict_prob() >= self.threshold





##########################################################################

##                             Run Program                              ##

##########################################################################


def main():

    # load the data
    iris = sklearn.datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1

    # visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # train the model
    model = LogisticRegression(X, y, learning_rate=0.1, iterations=300000, verbose=True)
    model.fit()
    preds = model.predict()
    print((preds == y).mean())
    print(model.theta)




if __name__ == "__main__":
    main()









# Unused



    # # Matrix solution of slope
    # from sklearn.datasets import load_boston
    # boston = load_boston()

    # # generate data
    # X = boston.data
    # y = boston.target
    # feature_names = boston.feature_names

    # # create vector of ones...
    # int = np.ones((len(y), 1))

    # #...and add to feature matrix
    # X = np.concatenate((int, X), 1)

    # # calculate coefficients using closed-form solution
    # coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    # feature_names = np.insert(boston.feature_names, 0, 'INT')
    # results = pd.DataFrame({'coefficients':coefficients}, index=feature_names)
    # print(results.round(2))








