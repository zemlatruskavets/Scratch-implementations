# To do:
# 1) Tidy up code





##############################################################################
##############################################################################
##############################################################################
######                                                                  ######
######                          DECISION TREES                          ######
######                                                                  ######
##############################################################################
##############################################################################
##############################################################################



##############################################################################

##      This code implements an algorithm to solve classification and       ##
##      regression problems using the decision tree approach.               ##

##############################################################################



"""

ALGORITHM STEPS

These are the steps that go into calculating the line of best fit:

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

# import random
import re
import numpy as np
import pandas as pd
# import seaborn as sns
import scipy.stats as sps
# import collections as col
# import matplotlib.pyplot as plt





##########################################################################

##                       User Defined Parameters                        ##

##########################################################################


# This is the name of the CSV file containing the data
dataset_name = 'creatures.csv'


# This dictionary defines the changes to be made to the initial dataframe
replacement_dictionary = {\
                          '?': np.nan,\
                          'good': 2\
                          }


# This contains the columns names of the features to be included in the model
relevant = ['toothed', 'hair', 'breathes', 'legs']


# This dictionary lists the normalization desired for each column
normalization_dictionary = {\
                              'Math':    'standardize',\
                              'Reading': 'standardize',\
                              'Label':   'standardize'\
                           }


# Specify whether each instance should be labelled by the first column
index = False

# Define the proportion of the data to be allocated as the test set
test_size = 0.2

# Define the range of parameters over which to test the model accuracy
parameter_range = range(0, 10, 1)


total_dictionary = {}



class Parse:

    """ 

    Data preparation for use in machine learning models.

        Attributes:     
            data: the dataset to be analyzed. 

        Methods:    
            ingest(): address missing or anomalous data
            clean(): parse the portion of the dataset of interest
            scale(): ensure data is within small range of values
            split(): separate dataset into training and test sets

        Usage:  
            >>> p = Parse('data.csv')
            
    """ 


    def __init__(self, data):
        self.data = data


    def ingest(self):
        """ Ensure compatible data types and address missing or anomalous entries """

        # Read first row and determine if job is supervised
        columns = pd.read_csv(self.data, nrows=0).columns
        if columns[-1] == 'Label':
            mode = 'Supervised'
            relevant.append('Label')
        else:
            mode = 'Unsupervised'

        # Read selected data into dataframe
        if index == True:
            dataframe = pd.read_csv(self.data, index_col=0, usecols=relevant)
        else:    
            dataframe = pd.read_csv(self.data, usecols=relevant)

        # Add a column to store the predictions
        dataframe['Prediction'] = 0

        # Remove 'Label' from relevant
        if 'Label' in relevant:
            relevant.remove('Label')


        return dataframe


    def clean(self, dataframe, replacement_dictionary, nan_replacement='mean'):
        """ Parse the relevant portion of the dataset """
        
        # Determine if data is labelled
        if dataframe.columns[-2] == 'Label':
            last_column = -2
        else:
            last_column = -1


        # Make all user-specified replacements in the dataframe
        if replacement_dictionary:
            for i in replacement_dictionary:
                dataframe.replace(i, replacement_dictionary[i], inplace=True)


        # Ensure all columns except the last are numeric       
        pd.options.mode.chained_assignment = None  # turn off SettingWithCopyWarning 
        dataframe[dataframe.columns[:last_column]] = dataframe[dataframe.columns[:last_column]]\
                                                     .apply(pd.to_numeric, downcast='float',\
                                                     errors='coerce', axis=1)


        # Replace NaNs with value based on others within column
        if nan_replacement == 'mean':
            dataframe = dataframe.fillna(dataframe.mean())
            
        if nan_replacement == 'median':
            dataframe = dataframe.fillna(dataframe.median())

        if nan_replacement == 'mode':
            dataframe = dataframe.fillna(dataframe.mode())

        if nan_replacement == 'min':
            dataframe = dataframe.fillna(dataframe.min())

        if nan_replacement == 'max':
            dataframe = dataframe.fillna(dataframe.max())

        if nan_replacement == 'zero':
            dataframe = dataframe.fillna(0)


        return dataframe



    def scale(self, data, mode='standardize'):
        """ Ensure numerical data is within small range of values """
        
        if mode == 'standardize':
            standardized = (data - data.mean()) / data.std()
            return standardized

        if mode == 'min-max':
            min_max = (data - data.min()) / (data.max() - data.min())
            return min_max

        if mode == 'mean':
            mean = (data - data.mean()) / (data.max() - data.min())
            return mean

        if mode == 'unit-length':
            unit = data / np.linalg.norm(data)
            return unit



    def split(self, data, split_proportion): 
        """ Split randomized dataset into training and test sets """

        data = data.sample(frac=1)

        training_data = data[int(split_proportion * len(data)):]
        test_data     = data[:int(split_proportion * len(data))]

        return training_data, test_data






class Analysis:

    """ 

    Data analysis for use in machine learning models.

        Attributes:     
            data: the parsed dataset to be analyzed
            k: the number of neighbours to include

        Methods:    
            gain(): calculates the gain for splitting on each feature
            split(): splits the dataset on the feature with the largest gain
            random_forest(): combines several decision trees
            purity(): calculates the purity of a given dataset

        Usage:  
            >>> a = Analysis(data, 4)    
            
    """ 

    def __init__(self, data):
        self.data = data




    def gain(dataframe):
        """ This function calculates the gain for splitting on each feature """

        gain_dictionary = {}

        for i in dataframe.columns[:-1]:

            # Determine the entropy of the labels
            label_prob    = dataframe['Label'].value_counts() / len(dataframe)
            label_entropy = sps.entropy(label_prob, base=2)
            
            # Determine the probabilities of the entries in the feature
            feature_prob = dataframe[i].value_counts() / len(dataframe)

            # Define dictionaries to store the probabilities and entropies
            probs    = {}
            entropy  = {}

            # Determine the entropy of the feature
            for j, k in enumerate(feature_prob.index):
                probs[k]     = dataframe['Label'][dataframe[i] == k].value_counts()
                probs[k]    /= len(dataframe['Label'][dataframe[i] == k])
                entropy[k]   = sps.entropy(probs[k], base=2) * feature_prob[j]

            feature_entropy = sum(entropy.values())

            # Determine the gain of splitting on this feature
            gain               = label_entropy - feature_entropy
            gain_dictionary[i] = gain


        return gain_dictionary




    def split(dataframe, gain_dictionary, index):
        """ The function splits the dataset on the feature with the largest gain """

        # Determine the feature with the largest gain
        feature = max(gain_dictionary, key=gain_dictionary.get)

        # Determine the unique values for this feature
        unique_values = dataframe[feature].unique()

        # Split the parent dataframe based on the values of this feature
        for i, j in enumerate(unique_values):
            number_entries = len(total_dictionary)

            total_dictionary['%i.%i' % (index + 2, number_entries + 1)] = \
            dataframe[dataframe[feature] == j].drop(columns=feature)



    def random_forest(dataframe):
        pass

        # Generate random subsets of overall dataset


        # Create decision trees for each subsets


        # Predict the labels for the unknown data with each tree


        # Perform a majority vote


        # Compare with the true labels


    def purity():
      """ Calculate the purity of a given dataset """





class Performance:

    """ 

    Methods for scoring models.

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


    def __init__(self, data, iterations):
        self.data = data
        self.iterations = iterations


    def score(self, test_data, training_data, mode=None):
        """ Determine accuracy of predictions on test data """

        wrong    = len(test_data[test_data['Label'] != test_data['Prediction']])
        right    = len(test_data) - wrong
        accuracy = (right / float(len(test_data)))


        # Print the final tally once all the test data has been parsed
        if mode == 'print':
            print("""
##########################################################################
##########################################################################
                                   
                                                                          
                            FINAL RESULTS

                      Total datapoints:\t\t\t\t%g                       
                      Training set size:\t\t\t%g
                      Test set size:\t\t\t\t%g 

                      Correct predictions:\t\t\t%g
                      Incorrect predictions:\t\t%g

                      Model accuracy:\t\t\t\t%.2f%%


##########################################################################
##########################################################################

          """ % (len(self.data), len(training_data), len(test_data), right, wrong, (accuracy * 100)))

        return accuracy



    def repeat_performance(self, iterations=10):
        """ Loop over parameter range and output accuracy """

        for k in parameter_range:
            score = 0
            for x in range(iterations):

                t  = Parse(dataset_name)

                training_data, test_data = t.split(data, test_size)
                a = Analysis(data)

                for i in test_data:
                    distances = a.distance(i, training_data, 2)
                    prediction = a.determine_class(distances, k)
                    z = a.score_prediction(distances, prediction, data, training_data, test_data, mode='printt')
                    if z != None:
                        score += z

            print('The score for %i neighbours is: %.2g%%' % (k, score * 10))



    def scikitlearn(self):
        """ Compare against scikit-learn """

        # Import the libraries
        from sklearn               import tree
        from sklearn.metrics       import mean_squared_error
        from sklearn.preprocessing import MinMaxScaler

        # Split the features and labels
        X = self.data[relevant]
        y = self.data['Label']

        # Scale the data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(X)

        # Fit the model
        skl_dt = tree.DecisionTreeClassifier()
        skl_dt.fit(X, y)

        # Determine the accuracy
        prediction = skl_dt.predict(X)

        # Determine the mean square error
        mse  = mean_squared_error(y, prediction)
        rmse = np.sqrt(mse)
        r2   = skl_dt.score(X, y)

        # Plot the data and regression line

        # return r2
        return rmse, r2





class Plotting:

    def __init__(self, data):
        self.data = data



    def pairplot(self):
        """ Use Seaborn to generate a pairplot """
        sns.set_style('white', {'font.family':'serif', 'font.serif':'Palatino'})
        colours = sns.color_palette("husl")
        sns.pairplot(self.data[relevant], hue=self.data['Prediction'], palette=colours)
        plt.show()



    def scatter_plot(self):
        """ Use Seaborn to generate scatterplot """
        sns.lmplot(x=relevant[0], y=relevant[1], data=self.data, fit_reg=False, \
            hue=self.data['Prediction'], palette="ch:.25", legend=False)
        plt.legend(loc='lower right')

        plt.show()
















##########################################################################

##                             Run Program                              ##

##########################################################################


def main():
    
    # 1) Parse and clean the data
    p         = Parse(dataset_name)
    dataframe = p.clean(p.ingest(), replacement_dictionary)
    a         = Analysis(dataframe)



    for i, j in enumerate(relevant):

        # Calculate the gain for each feature
        gain_dictionary = gain(dataframe)

        # Split the tree based on the feature with the largest gain
        split(dataframe, gain_dictionary, i)


        # Repeat for all split nodes
        for k in total_dictionary:
            if re.match(r'%i.' % (i + 2), k):

                # Calculate the gain for each feature
                gain_dictionary = gain(total_dictionary[k])

                # Split the tree based on the feature with the largest gain
                split(total_dictionary[k], gain_dictionary, i + 1)




    # Working        
    # gain_dictionary = gain(dataframe)

    # split(dataframe, gain_dictionary, 0)

    # print(total_dictionary)

    # for i in total_dictionary:
    #     if re.match(r'%i.' % 2, i):
    #         print(i)
 










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



    # dataframe = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
    #                           "hair":["True","True","False","True","True","True","False","False","True","False"],
    #                           "breathes":["True","True","True","True","True","True","False","True","True","True"],
    #                           "legs":["True","True","False","True","True","True","False","False","True","True"],
    #                           "Label":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]}, 
    #                         columns=["toothed","hair","breathes","legs","Label"])




