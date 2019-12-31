# -*- coding: utf-8 -*-


# To do:
# 1) Implement cross-entropy loss function
# 2) Fill out activation function method





##############################################################################
##############################################################################
##############################################################################
######                                                                  ######
######                     ARTIFICIAL NEURAL NETWORK                    ######
######                                                                  ######
##############################################################################
##############################################################################
##############################################################################



##############################################################################

##      This code implements an algorithm to solve regression               ##
##      problems using an artificial neural network.                        ##

##############################################################################



"""

ALGORITHM STEPS

These are the steps that go into an artificial neural network:

    1) Parse the data.
    2) Split the data into training and test data.
    3) Normalize the training data.
    4) Separate the features and labels for the datasets.
    5) Randomly initialize the weight matrices for each layer.
    6) Perform the feedforward pass:
        1) Initialize the dictionaries to store the results.
        2) Take the matrix product of the activations and weights and add the bias.
        3) Apply the specified activation function to the product.
        4) Store the activations and derivatives.
        5) Repeat steps 1) - 4) for each layer.
    7) Perform the backwards propagation pass:
        1) Determine the difference between the predicted and given values.
        2) Determine the delta terms.
        2) Determine the partial derivative for each parameter.
        3) Input the values into these expressions as necessary.
        4) Form the Jacobian matrix for each layer.
        5) Subtract the scaled Jacobian from each weight matrix.
    8) Adjust the weight and bias matrices.
    9) Repeat steps 6) - 8).
    10) Determine the model accuracy using the test data.
    

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
import numpy as np
import pandas as pd
# import seaborn as sns
# import tensorflow as tf
import matplotlib.pyplot as plt





##########################################################################

##                       User Defined Parameters                        ##

##########################################################################


# This is the name of the CSV file containing the data
dataset_name = 'car_features.csv'


# This dictionary defines the changes to be made to the initial dataframe
replacement_dictionary = {\
                          '?': np.nan,\
                          'Diesel': -1,\
                          'Essence': 1,\
                          'GPL': 1,\
                          'Autre': 1\
                          }


# This contains the columns names of the features to be included in the model
relevant = ['km', 'fuel', 'age']


# This dictionary lists the normalization desired for each column
normalization_dictionary = {\
                              'km':    'standardize',\
                              'age':   'standardize',\
                              'Label': 'min-max',\
                           }

activation_function = 'relu'

# Specify whether each instance should be labelled by the first column
index = False

# Define the proportion of the data to be allocated as the test set
test_size = 0.2

# Specify the number of activations in the hidden and output layers
network_architecture = [len(relevant), 3, 2, 1]

iterations = 1000

Lambda = 0.01


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
                                                     .apply(pd.to_numeric, errors='coerce', axis=1)


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






class Neural:


    def __init__(self, data, network_architecture, learning_rate, Lambda):
        self.data = data
        self.network_architecture = network_architecture
        self.learning_rate        = learning_rate
        self.Lambda               = Lambda



    def initialization(self, mode='Random'):
        """ Initialize the weight matrices """

        # Store the weight matrices in a dictionary
        weight_dictionary = {}
        bias_dictionary   = {}

        # Randomly initialize the weight matrices
        if mode == 'Random':
            for i, j in enumerate(self.network_architecture):
                if (i + 1) < len(self.network_architecture):
                    weight_dictionary[i + 1] = np.random.rand(j, self.network_architecture[i + 1])
                    bias_dictionary[i + 2]   = np.random.rand(1, self.network_architecture[i + 1])

        return weight_dictionary, bias_dictionary



    def forward_propagation(self, input_matrix, weight_dictionary, bias_dictionary, activation_function):
        # 1) Initialize the dictionaries to store the results.
        # 2) Take the matrix product of the activations and weights and add the bias.
        # 3) Apply the specified activation function to the product.
        # 4) Store the activations and derivatives.
        # 5) Repeat steps 1) - 4) for each layer.


        # 1) Initialize a dictionary to store all of the activations
        output_dictionary     = {}
        activation_dictionary = {}
        derivative_dictionary = {}

        # Seed the activation dictionary with the input data
        activation_dictionary[1] = input_matrix.values

        # 5) Repeat steps 2) - 4) for each layer
        for i in range(1, len(self.network_architecture)):

            # 2) Take the matrix product of the activations and weights and add the bias
            output_dictionary[i + 1] = np.dot(activation_dictionary[i], weight_dictionary[i]) + bias_dictionary[i + 1]

            # 3) Apply the specified activation function to the product          
            result, derivative = self.activation_function(output_dictionary[i + 1], activation_function)

            # 4) Store the activation and derivative
            activation_dictionary[i + 1] = result
            derivative_dictionary[i + 1] = derivative


        return output_dictionary, activation_dictionary, derivative_dictionary





    def back_propagation(self, y_train, weight_matrices, output_dictionary, activation_dictionary, derivative_dictionary):

        # 1) Determine the difference between the predicted and given values.
        # 2) Determine the delta terms.
        # 2) Determine the partial derivative for each parameter.
        # 3) Input the values into these expressions as necessary.
        # 4) Form the Jacobian matrix for each layer.
        # 5) Subtract the scaled Jacobian from each weight matrix.


        # Initialize the dictionaries
        delta_dictionary    = {}
        Jacobian_dictionary = {}

        # 1) Determine the difference between the predicted and given values
        y_train = y_train.values.reshape(len(y_train), 1)
        diff    = -(y_train - activation_dictionary[len(activation_dictionary)]) / len(self.data)

        # 2) Determine the delta terms
        for i in range(len(self.network_architecture), 1, -1):
            if i == len(self.network_architecture):
                delta_dictionary[i] = diff * derivative_dictionary[i]
            else:
                delta_dictionary[i] = np.dot(delta_dictionary[i + 1], weight_matrices[i].T) * derivative_dictionary[i]

        # 3) Determine the gradients for the weights
        for i in range(len(self.network_architecture) - 1, 0, -1):
            Jacobian_dictionary[i] = np.dot(activation_dictionary[i].T, delta_dictionary[i + 1])



        return delta_dictionary, Jacobian_dictionary





    def update(self, weight_dictionary, bias_dictionary, delta_dictionary, Jacobian_dictionary):
        """ Update the weights and biases """

        # 1) Update the weights
        for i in range(1, len(self.network_architecture)):
            regularization = self.Lambda * weight_dictionary[i]
            weight_dictionary[i] -= self.learning_rate * (Jacobian_dictionary[i] - regularization)

        # 2) Update the biases
        for i in range(2, len(self.network_architecture) + 1):
          bias_dictionary[i] -= self.learning_rate * np.sum(delta_dictionary[i], axis=0) 


        return weight_dictionary, bias_dictionary




    def activation_function(self, values, function='relu'):

        if function == 'identity':
            result     = values
            derivative = 1

            return result, derivative


        elif function == 'Heaviside':
            """ Heaviside step function """
            result = np.copy(values)
            result[result < 0] = 0
            result[result >= 0]  = 1           

            # Calculate the derivative
            derivative = np.zeros(result.shape)

            return result, derivative


        elif function == 'relu':
            """ rectified linear unit """
         
            result = np.maximum(values, 0)
            
            # Calculate the derivative
            derivative = np.copy(result)
            derivative[derivative <= 0] = 0
            derivative[derivative > 0]  = 1

            return result, derivative


        elif function == 'sigmoid':
            result     = 1 / (1 + np.exp(-values))
            derivative = result * (1 - result)

            return result, derivative


        elif function == 'arc-tan':
            result     = np.arctan(values)
            derivative = 1 / (1 + np.square(values))

            return result, derivative


        elif function == 'tanh': 
            result     = np.tanh(values)
            derivative = 1.0 - np.square(values)
            
            return result, derivative


        elif function == 'soft-plus':
            result     = np.log(1.0 + np.exp(values))
            derivative = 1 / (1 + np.exp(-values))
            
            return result, derivative


        elif function == 'sinusoid':
            result     = np.sin(values)
            derivative = np.cos(values)
            
            return result, derivative


        elif function == 'gaussian':
            result     = np.exp(-np.square(values))
            derivative = -2 * values * np.exp(-np.square(values))
            
            return result, derivative



    def prediction(self, test_data, weight_dictionary, bias_dictionary, activation_function):
        """ Determine the output of the trained network for the test data """

        # 1) Initialize a dictionary to store all of the activations
        output_dictionary     = {}
        activation_dictionary = {}
        derivative_dictionary = {}

        # Seed the activation dictionary with the input data
        activation_dictionary[1] = test_data.values

        # 5) Repeat steps 2) - 4) for each layer
        for i in range(1, len(self.network_architecture)):

            # 2) Take the matrix product of the activations and weights and add the bias
            output_dictionary[i + 1] = np.dot(activation_dictionary[i], weight_dictionary[i]) + bias_dictionary[i + 1]

            # 3) Apply the specified activation function to the product          
            result, derivative = self.activation_function(output_dictionary[i + 1], activation_function)

            # 4) Store the activation and derivative
            activation_dictionary[i + 1] = result
            derivative_dictionary[i + 1] = derivative


        return output_dictionary, activation_dictionary, derivative_dictionary





    def tensorflow(self, X, Y):

        # regularization strength
        Lambda = 0.01
        learning_rate = 0.01

        with tf.name_scope('input'):
            # training data
            x = tf.placeholder("float", name="cars")
            y = tf.placeholder("float", name="prices")

        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.random_normal([3, 3]), name="W1")
            w2 = tf.Variable(tf.random_normal([3, 2]), name="W2")
            w3 = tf.Variable(tf.random_normal([2, 1]), name="W3")

        with tf.name_scope('biases'):
            # biases (we separate them from the weights because it is easier to do that when using TensorFlow)
            b1 = tf.Variable(tf.random_normal([1, 3]), name="b1")
            b2 = tf.Variable(tf.random_normal([1, 2]), name="b2")
            b3 = tf.Variable(tf.random_normal([1, 1]), name="b3")

        with tf.name_scope('layer_1'):
            # three hidden layer
            layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, w1), b1))

        with tf.name_scope('layer_2'):
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, w2), b2))

        with tf.name_scope('layer_3'):
            layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3), b3))

        with tf.name_scope('regularization'):
            # L2 regularization applied on each weight
            regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

        with tf.name_scope('loss'):
            # loss function + regularization value
            loss = tf.reduce_mean(tf.square(layer_3 - y)) + Lambda * regularization
            loss = tf.Print(loss, [loss], "loss")

        with tf.name_scope('train'):
            # we'll use gradient descent as optimization algorithm
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # launching the previously defined model begins here
        init = tf.global_variables_initializer()


        with tf.Session() as session:
            session.run(init)

            # we'll make 5000 gradient descent iteration
            for i in range(10000):
                session.run(train_op, feed_dict={x: X, y: Y})

            builder.add_meta_graph_and_variables(session, ["dnn_from_scratch_tensorflow"])

            # testing the network
            print("Testing data")
            print("Loss: " + str(session.run([loss], feed_dict={x: X, y: Y})[0]))

            # do a forward pass
            print("Predicted price: " + str(predict.output(session.run(layer_3,
                                                                       feed_dict={x: predict.input(168000, "Diesel", 5)}))))





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





##########################################################################

##                             Run Program                              ##

##########################################################################


def main():
    
    # 1) Parse and clean the data
    p         = Parse(dataset_name)
    dataframe = p.clean(p.ingest(), replacement_dictionary)

    # 2) Split the dataframe into training and test data
    training_data, test_data = p.split(dataframe, test_size)

    # 3) Normalize the training data
    for i in normalization_dictionary:
        training_data[i] = p.scale(training_data[i], mode=normalization_dictionary[i])

    # 4) Separate the features and labels for the datasets
    X_train, y_train = training_data[relevant], training_data['Label']
    X_test, y_test  = test_data[relevant], test_data['Label']

    # 5) Randomly initialize the weight matrices for each layer.
    nn     = Neural(X_train, network_architecture, 1e-3, 1e-2)
    wm, bm = nn.initialization()

    # 9) Train the network (repeat steps 6) - 8))
    for i in range(iterations):
        
        # 6) Perform the feedforward pass:
        output_dictionary,\
        activation_dictionary,\
        derivative_dictionary = nn.forward_propagation(X_train, wm, bm, activation_function)

        # 7) Perform the backwards propagation pass:
        delta_dictionary, Jacobian_dictionary = \
        nn.back_propagation(y_train, wm, output_dictionary, activation_dictionary, derivative_dictionary)

        # 8) Adjust the weight and bias matrices
        wm, bm = nn.update(wm, bm, delta_dictionary, Jacobian_dictionary)


    # 10) Determine the model accuracy
    prediction = nn.forward_propagation(X_test, wm, bm, activation_function)[1][len(network_architecture)]


    predict_df = pd.concat([X_test, y_test], axis=1)
    predict_df['Prediction'] = prediction
    predict_df['Error'] = (np.abs(predict_df['Label'] - predict_df['Prediction']) / predict_df['Label']) * 100

    print(predict_df)









if __name__ == "__main__":
    main()

















        #     if mode == 'initial':
        #         # 1) Add a column of 1s to activation matrix
        #         ones = np.ones(shape=(len(activation_dictionary['a%i' % i]), 1))
        #         activation_dictionary['a%i' % i] = np.hstack((activation_dictionary['a%i' % i], ones))  

        #         # Add a row corresponding to the bias to the weight matrix
        #         row = np.ones(shape=weight_matrices['w%i' % i].shape[1])
        #         weight_matrices['w%i' % i] = np.vstack([weight_matrices['w%i' % i], row])         

        #     # 3) Multiply the feature and weight matrices
        #     activation = np.dot(activation_dictionary['a%i' % i], weight_matrices['w%i' % i])

        #     # 4) Apply the specified activation function to the product          
        #     value, gradient = self.activation_function(activation, function='tanh')
        #     activation_dictionary['a%i' % (i + 1)] = value
        #     derivative_dictionary['d%i' % (i + 1)] = gradient

        #     # # 1) Add a column of 1s to derivative matrix
        #     # ones = np.ones(shape=(len(derivative_dictionary['d%i' % (i + 1)]), 1))
        #     # derivative_dictionary['d%i' % (i + 1)] = np.hstack((derivative_dictionary['d%i' % (i + 1)], ones))  

# input_matrix['Bias'] = np.ones(len(input_matrix))
# weight_matrices[i].loc[weight_matrices[i].shape[0]] = np.ones(weight_matrices[i].shape[1])

        # return activation_dictionary, weight_matrices, derivative_dictionary




    # def forward_propagation(self, feature_matrix, weight_matrices, mode='initial'):
    #     # 1) Initialize the dictionaries to store the results.
    #     # 2) Add a column of 1s to the input matrix.
    #     # 3) Add the bias array as a row to the weight matrix.
    #     # 4) Multiply the feature and weight matrices.
    #     # 5) Apply the specified activation function to the product.
    #     # 6) Store the activations and derivatives.
    #     # 7) Repeat steps 1) - 4) for each layer.


    #     # 1) Initialize a dictionary to store all of the activations
    #     output_dictionary     = {}
    #     activation_dictionary = {}
    #     derivative_dictionary = {}

    #     # Seed the activation dictionary with the input data
    #     activation_dictionary[1] = feature_matrix.values


    #     # 7) Repeat steps 2) - 6) for each layer
    #     for i in range(1, len(self.network_architecture)):

    #         # 2) Add a column of 1s to the input matrix
    #         input_activation = activation_dictionary[i]
    #         input_activation = np.hstack((input_activation, np.ones(shape=(len(input_activation), 1))))

    #         # 3) Add the bias array as a row to the weight matrix
    #         input_weight = weight_matrices[i]
    #         input_weight = np.vstack([input_weight, np.ones(shape=input_weight.shape[1])])                     

    #         # 4) Multiply the feature and weight matrices
    #         output_dictionary[i + 1] = np.dot(input_activation, input_weight)

    #         # 5) Apply the specified activation function to the product          
    #         result, derivative = self.activation_function(output_dictionary[i + 1], 'tanh')

    #         # 6) Store the activation and derivative
    #         activation_dictionary[i + 1] = result
    #         derivative_dictionary[i + 1] = derivative


    #     return weight_matrices, output_dictionary, activation_dictionary, derivative_dictionary






