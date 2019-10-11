# To do:
# 1) Implement weighted distance voting





##############################################################################
##############################################################################
##############################################################################
######                                                                  ######
######                       K-NEAREST NEIGHBOURS                       ######
######                                                                  ######
##############################################################################
##############################################################################
##############################################################################



##############################################################################

##      This code implements an algorithm to solve classification           ##
##      problems using the k-nearest neighbours approach.                   ##

##############################################################################



"""

ALGORITHM STEPS

These are the steps that go into the k-nearest neighbours algorithm:

    1) Normalize the data.
    2) Split the dataset into training and test sets.
    3) Calculate the distances between all pairs of points in the test 
       and training sets.
    4) Determine the k closest points for each test point.
    5) Perform a majority vote for the closest points for each point in 
       the test set, performing a tie break, if necessary.
    6) Calculate the accuracy of the algorithm by comparing the predicted
       classifications with the true classifications.
    7) Plot the data.


DATASET ASSUMPTIONS

These are the assumptions made about the input dataset:
    
    1) It is a .csv file.
    2) It is a 'tidy' dataset (look it up).
    3) You know exactly how you want to deal with anomalous entries.
    4) If it is a supervised problem, the last column is called 'Class'.
    5) No columns are called 'Class' or 'Prediction'.


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





##########################################################################

##                       User Defined Parameters                        ##

##########################################################################


# This is the name of the CSV file containing the data
dataset_name = 'kidney_disease.csv'


# This dictionary defines the changes to be made to the initial dataframe
replacement_dictionary = {\
                          '?': np.nan,\
                          'yes': 4,\
                          'no': 2,\
                          'notpresent': 4,\
                          'present': 2,\
                          'abnormal': 4,\
                          'normal': 2,\
                          'poor': 4,\
                          'good': 2\
                          }


# This contains the columns names of the features to be included in the model
relevant = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane']


# This dictionary lists the normalization desired for each column
normalization_dictionary = {\
                              'age':   'standardize',\
                              'bp':    'standardize',\
                              'sg':    'standardize',\
                              'al':    'standardize',\
                              'su':    'standardize',\
                              'rbc':   'standardize',\
                              'pc':    'standardize',\
                              'pcc':   'standardize',\
                              'ba':    'standardize',\
                              'bgr':   'standardize',\
                              'bu':    'standardize',\
                              'sc':    'standardize',\
                              'sod':   'standardize',\
                              'pot':   'standardize',\
                              'hemo':  'standardize',\
                              'pcv':   'standardize',\
                              'wbcc':  'standardize',\
                              'rbcc':  'standardize',\
                              'htn':   'standardize',\
                              'dm':    'standardize',\
                              'cad':   'standardize',\
                              'appet': 'standardize',\
                              'pe':    'standardize',\
                              'ane':   'standardize'\
                           }



# Specify whether each instance should be labelled by the first column
index = False

# Define the proportion of the data to be allocated as the test set
test_size = 0.2

# Define the number of clusters
k_clusters = 4

# Define the range of parameters over which to test the model accuracy
parameter_range = range(0, 10, 1)

# Define the threshold governing convergence
threshold = 0.01

# Specify the number of nearest neighbours
knn = 5



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
        if columns[-1] == 'Class':
            mode = 'Supervised'
            relevant.append('Class')
        else:
            mode = 'Unsupervised'

        # Read selected data into dataframe
        if index == True:
            dataframe = pd.read_csv(self.data, index_col=0, usecols=relevant)
        else:    
            dataframe = pd.read_csv(self.data, usecols=relevant)

        # Add a column to store the predictions
        dataframe['Prediction'] = 0

        # Remove 'Class' from relevant
        if 'Class' in relevant:
            relevant.remove('Class')


        return dataframe


    def clean(self, dataframe, replacement_dictionary, nan_replacement='mean'):
        """ Parse the relevant portion of the dataset """
        
        # Determine if data is labelled
        if dataframe.columns[-2] == 'Class':
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
            distance(): calculate distance between points
            determine_class(): find k-nearest neighbours and take majority vote
            tie_breaker(): specify prediction in event of a tie
            score_prediction(): determine accuracy of model

        Usage:  
            >>> a = Analysis(data, 4)    
            
    """ 

    def __init__(self, data):
        self.data    = data
        self.correct = 0
        self.false   = 0


    # def initialize_positions(self, k_clusters, data, parsed, mode='noise'):
    def initialize_positions(self, k_clusters, mode='noise'):
        """ Return an array of random initial points """
        
        points_list = []

        # This loop takes a random point and then adds noise to each point
        for i in range(k_clusters):

            # This selects random datapoints and adds noise to them
            if mode == 'noise':
                random_point = self.data.iloc[np.random.randint(0, len(self.data))].values

                for j, k in enumerate(relevant):
                    random_point[j] += np.random.normal(loc=0, scale=self.data[k].std()/2)

            # This designates points distributed according to a normal distribution
            if mode == 'normal':
                random_point = np.zeros(len(relevant))
                for j, k in enumerate(relevant):
                    random_point[j] += np.random.normal(loc=self.data[k].mean(), scale=self.data[k].std())

            points_list.append(random_point)

        return np.array(points_list)



    def distance(self, test, training, p, special=None):
        """ Calculate the Minkowski distances between test point and points in training set """

        # Calculate the distance between the test point and all points in the training set
        if special == 'Manhattan':  
            distances = np.absolute(test - training).sum(axis=1)
        else:
            distances = np.power((np.power(test - training, p)).sum(axis=1), (1 / float(p)))
        
        return distances



    def determine_class(self, distances, k):
        """ Determine the k-nearest neighbours to a given point """

        # To do:
        # 1) Learn how to determine most common value in dataframe column


        # Sort array from least to greatest distances and take k lowest distances
        sorted_array = distances[np.argsort(distances[:, 0])][:k]

        # Sum the number of instances of the classes in the lowest k distances
        neighbours = col.Counter(sorted_array[:, -2])

        # Determine the most common class
        prediction = neighbours.most_common()[0][0]

        return prediction



    def tie_breaker(self, distances):
        """ Determine class in the event of tie in majority vote """

        # To do:
        # 1) Write segment that determines closest distance
        # 2) Write segment that determines closest average distance


    def score_prediction(self, test_data, training_data, mode=None):
        """ Determine accuracy of predictions on test data """

        wrong    = len(test_data[test_data['Class'] != test_data['Prediction']])
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



class Plotting:

    def __init__(self, data):
        self.data = data



    def pairplot(self, dataframe, classname):
        """ Use Seaborn to generate a pairplot """
        sns.set_style('white', {'font.family':'serif', 'font.serif':'Palatino'})
        colours = sns.color_palette("husl")
        sns.pairplot(dataframe, hue=classname, palette=colours)
        plt.show()



    def scatter_plot(self, relevant, dataframe, classname):
        """ Use Seaborn to generate scatterplot """
        sns.lmplot(x=relevant[0], y=relevant[1], data=dataframe, fit_reg=False, \
            hue=classname, palette="ch:.25", legend=False)
        plt.legend(loc='lower right')

        plt.show()





class Performance:
    def __init__(self, parameter_range, iterations):
        self.iterations = iterations


    def repeat_performance(self, iterations=10):
        """ Loop over parameter range and output accuracy """

        for k in parameter_range:
            score = 0
            for x in range(iterations):

                t  = Parse(dataset_name)
                df = t.ingest(replacement_dictionary)
                classname = t.clean(df, relevant)[0]
                parsed    = t.clean(df, relevant)[1]
                data      = t.clean(df, relevant)[2]
                training_data, test_data = t.split(data, test_size)
                a = Analysis(data)

                for i in test_data:
                    distances = a.distance(i, training_data, 2)
                    prediction = a.determine_class(distances, k)
                    z = a.score_prediction(distances, prediction, data, training_data, test_data, mode='printt')
                    if z != None:
                        score += z

            print('The score for %i neighbours is: %.2g%%' % (k, score * 10))




    def scikitlearn(self, data, testsize, neighbours):
        """ Compare against scikit-learn """

        # Import the libraries
        from sklearn.cross_validation import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        # Split the features and labels
        X = data[relevant]
        y = data['Class']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=42)

        # Fit the model
        skl_knn = KNeighborsClassifier(n_neighbors=neighbours)
        skl_knn.fit(X_train, y_train)

        # Determine the accuracy
        pred = skl_knn.predict(X_test)
        print(accuracy_score(y_test, pred))



##########################################################################

##                             Run Program                              ##

##########################################################################


def main():
    
    # 1) Parse and clean the data
    p         = Parse(dataset_name)
    dataframe = p.clean(p.ingest(), replacement_dictionary, nan_replacement='mean')
    a         = Analysis(dataframe)


    # 2) Normalize the data
    for i in normalization_dictionary:
        dataframe[i] = p.scale(dataframe[i], mode=normalization_dictionary[i])


    # 3) Split the dataframe into training and test data
    training_data, test_data = p.split(dataframe, test_size)


    # 4) For each training point, determine k closest neighbours
    for i in range(len(test_data)):

        # Make a copy of the test set dataframe
        temp_df = training_data
        point   = test_data.iloc[i][relevant].values
        index   = test_data.iloc[i][relevant].name

        # Fill in column with distances to all points in training set
        temp_df['Distance'] = a.distance(point, temp_df[relevant].values, 2)
        temp_df = temp_df.sort_values(by='Distance')

        # Determine k closest neighbours and add to prediction column
        prediction = temp_df['Class'].iloc[:knn].value_counts().idxmax()
        test_data['Prediction'][index] = prediction
   

    # Determine accuracy
    a.score_prediction(test_data, training_data, mode='print')



if __name__ == "__main__":
    main()















