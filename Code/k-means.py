# To do:
# 1) Implement k++ in initialization





##############################################################################
##############################################################################
##############################################################################
######                                                                  ######
######                              K-MEANS                             ######
######                                                                  ######
##############################################################################
##############################################################################
##############################################################################



##############################################################################

##      This code implements an algorithm to solve classification           ##
##      problems using the k-means approach.                                ##

##############################################################################



"""

ALGORITHM STEPS

These are the steps that go into the k-means algorithm:

    1) Parse the data.
    2) Normalize the data.
    3) Randomly distribute centre points throughout the phase space.
    4) Plot the data.
    5) Calculate the distances between all points in the dataset and each of 
       these centre points.
    6) For all points in the dataset, determine the closest centre point.
    7) Calculate the 'centre of mass' for each of the resulting clusters.
    8) Move the centre point for each cluster to its centre of mass.
    9) Repeat the process from steps 5) to 8)
    10) Stop when the movement in step 8) is below a predefined threshold.    
    11) Plot the data.


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
dataset_name = 'Bit9.csv'
# dataset_name = 'cricket.csv'


# This dictionary defines the changes to be made to the initial dataframe
replacement_dictionary = {\
                          '?': np.nan\
                          }


# This contains the columns names of the features to be included in the model
relevant = ['Count', 'Unique count of name', 'Unique count of process']
# relevant = ['batting', 'bowling']



# This dictionary lists the normalization desired for each column
normalization_dictionary = {\
                              'Count':                   'standardize',\
                              'Unique count of name':    'standardize',\
                              'Unique count of process': 'standardize'\
                           }



# Specify whether each instance should be labelled by the first column
index = True

# Define the proportion of the data to be allocated as the test set
test_size = 0.2

# Define the number of clusters
k_clusters = 2

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
            dataframe = pd.read_csv(self.data, index_col=0)
            dataframe = dataframe[relevant]
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
        
        data = self.data[relevant]
        points_list = []

        # This loop takes a random point and then adds noise to each point
        for i in range(k_clusters):

            # This selects random datapoints and adds noise to them
            if mode == 'noise':
                random_point = data.iloc[np.random.randint(0, len(data))].values

                for j, k in enumerate(relevant):
                    random_point[j] += np.random.normal(loc=0, scale=data[k].std()/2)

            # This designates points distributed according to a normal distribution
            if mode == 'normal':
                random_point = np.zeros(len(relevant))
                for j, k in enumerate(relevant):
                    random_point[j] += np.random.normal(loc=data[k].mean(), scale=data[k].std())
            
            # This initializes centroids based on the k++ algorithm
            # if mode == 'k++':
            #     if i == 0:
            #         random_point = np.zeros(len(relevant))
            #         for j, k in enumerate(relevant):
            #             random_point[j] += np.random.normal(loc=dataframe[k].mean(), scale=dataframe[k].std())
            

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
        # 1) Write segment that determines closest distance
        # 2) Write segment that determines closest average distance



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



    def pairplot(self):
        """ Use Seaborn to generate a pairplot """
        sns.set_style('white', {'font.family':'serif', 'font.serif':'Palatino'})
        colours = sns.color_palette("husl")
        print(self.data[relevant])
        # sns.pairplot(self.data[relevant], hue=self.data['Prediction'], palette=colours)
        # plt.show()



    def scatter_plot(self):
        """ Use Seaborn to generate scatterplot """
        sns.lmplot(x=relevant[0], y=relevant[1], data=self.data, fit_reg=False, \
            hue=self.data['Prediction'], palette="ch:.25", legend=False)
        plt.legend(loc='lower right')

        plt.show()





class Performance:
    def __init__(self, iterations):
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




    def scikitlearn(self, data, clusters):
        """ Compare against scikit-learn """

        # Import the libraries
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler

        # Split the features and labels
        data = data[relevant]

        # Scale the data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Fit the model
        skl_km = KMeans(n_clusters=clusters)
        skl_km.fit(data_scaled)

        # Determine the accuracy
        prediction = skl_km.predict(data_scaled)

        # Plot resulting clusters  
        plt.scatter(data[data.columns[0]],data[data.columns[1]], c=skl_km.labels_, cmap='rainbow')  
        plt.scatter(skl_km.cluster_centers_[:,0] ,skl_km.cluster_centers_[:,1], color='black')  
        plt.show()



##########################################################################

##                             Run Program                              ##

##########################################################################


def main():
    

    # 1) Parse and clean the data
    p         = Parse(dataset_name)
    dataframe = p.clean(p.ingest(), replacement_dictionary)


    # 2) Normalize the data
    for i in normalization_dictionary:
        dataframe[i] = p.scale(dataframe[i], mode=normalization_dictionary[i])


    # 3) Randomly distribute centre points throughout the phase space
    a = Analysis(dataframe)
    centroids = a.initialize_positions(k_clusters, mode='noise')


    # 4) Plot the data
    # sns.regplot(x=dataframe[relevant[0]], y=dataframe[relevant[1]], fit_reg=False)
    
    # Plot 3 dimensional data
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataframe[relevant[0]], dataframe[relevant[1]], dataframe[relevant[2]])
    plt.show()


    # 9) Repeat the process from steps 5) to 8)
    for iteration in range(10):

        # Determine the distance between the old and new centroids
        if iteration > 0:

            # 10) Stop when the movement in step 8) is below a predefined threshold
            if np.max(a.distance(old_centroids, centroids, 2)) < threshold:
                print('The centroids have converged to less than the threshold in %i iterations.'\
                     % (iteration + 1))
                print('Their positions are:')
                for i, j in enumerate(centroids):
                    print('Class %i: %s' % (i+1, j))
                print('The final SSE for %i clusters is: %g' % (k_clusters, sse))
                break
            old_centroids = centroids

        else:
            old_centroids = centroids

        
        # Define the lists to hold the coordinates and centroids
        coordinate_list   = []
        updated_centroids = []


        # Determine the closest centroid for each datapoint
        for i in range(len(dataframe)):
            point = dataframe.iloc[i][relevant].values
            index = np.argmin(a.distance(point, centroids, 2))
            dataframe['Prediction'].iloc[i] = index


        # Determine the centre-of-mass for each cluster
        for i in range(len(centroids)):
            separated = dataframe.loc[dataframe['Prediction'] == i]
            updated_centroids.append(separated[relevant].mean().values)
        centroids = np.array(updated_centroids)


        # Determine the summed square of errors
        sse = 0
        for i, j in enumerate(centroids):
            data = dataframe[relevant].loc[dataframe['Prediction'] == i].values
            sse += np.sum(np.square(a.distance(j, data, 2)))


        # 11) Plot the data
        # plot = sns.lmplot( x=relevant[0], y=relevant[1], data=dataframe, fit_reg=False, hue='Prediction', legend=False)

        # Plot 3 dimensional data
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(dataframe[relevant[0]], dataframe[relevant[1]], dataframe[relevant[2]], c=dataframe['Prediction'])
        
        # plot.savefig('output-%i.png' % iteration)
        plt.show()




    # print(dataframe['Prediction'])
    # pp = Plotting(dataframe)
    # pp.scatter_plot()

    # pp = Performance(3)
    # pp.scikitlearn(dataframe, 3)


if __name__ == "__main__":
    main()










# Unused

# relevant = ['batting', 'bowling']
# relevant = ['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']


# replacement_dictionary = {\
#                           '?': np.nan,\
#                           'yes': 4,\
#                           'no': 2,\
#                           'notpresent': 4,\
#                           'present': 2,\
#                           'abnormal': 4,\
#                           'normal': 2,\
#                           'poor': 4,\
#                           'good': 2\
#                           }



# normalization_dictionary = {\
#                               'age':   'standardize',\
#                               'bp':    'standardize',\
#                               'sg':    'standardize',\
#                               'al':    'standardize',\
#                               'su':    'standardize',\
#                               'rbc':   'standardize',\
#                               'pc':    'standardize',\
#                               'pcc':   'standardize',\
#                               'ba':    'standardize',\
#                               'bgr':   'standardize',\
#                               'bu':    'standardize',\
#                               'sc':    'standardize',\
#                               'sod':   'standardize',\
#                               'pot':   'standardize',\
#                               'hemo':  'standardize',\
#                               'pcv':   'standardize',\
#                               'wbcc':  'standardize',\
#                               'rbcc':  'standardize',\
#                               'htn':   'standardize',\
#                               'dm':    'standardize',\
#                               'cad':   'standardize',\
#                               'appet': 'standardize',\
#                               'pe':    'standardize',\
#                               'ane':   'standardize'\
#                            }












