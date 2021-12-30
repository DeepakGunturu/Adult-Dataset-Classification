# Name: Deepak Kumar Gunturu
# FIDN: A18672252
# Final Project: Multi-layer neural networks and SVM for Adult Dataset classification

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import timeit
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Cleaning the datasets
def cleanDataset(trainData,testData):
    # Cleaning training data    
    trainData = trainData.drop_duplicates()
    trainData = trainData.dropna()
    trainData['income'] = trainData['income'].map({' <=50K':0, ' >50K':1})
    
    # Cleaning test data
    testData = testData.drop_duplicates()
    testData = testData.dropna()
    testData['income'] = testData.income.str.rstrip('.')
    testData['income'] = testData['income'].map({' <=50K':0, ' >50K':1})
       
    return trainData, testData

# Preprocessing the datasets
def preprocess(trainData,testData):
    
    # Training and testing datasets split into features and labels
    X_train = trainData.drop(['income'],axis = 1)
    y_train = trainData['income']

    X_test = testData.drop(['income'],axis = 1)
    y_test = testData['income']

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    Y_train = []
    
    # Encoding the training labels
    for i in y_train:
        Y_train.append([1-i,i])

    Y_test = []

    # Encoding the test labels
    for i in y_test:
        Y_test.append([1-i,i])

    Y_train1 = np.array(Y_train)
    Y_test1 = np.array(Y_test)

    # Min-max scaling done on continuous features on training and testing datasets
    X_train[['Age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']] = MinMaxScaler().fit_transform(X_train[['Age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']])
    X_test[['Age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']] = MinMaxScaler().fit_transform(X_test[['Age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']])

    # One-hot encoding on training and testing categorical features 
    X_train = pd.get_dummies(trainData,columns = ['Workclass','education','marital-status','occupation','relationship','race','gender','native-country'])
    X_test = pd.get_dummies(testData,columns = ['Workclass','education','marital-status','occupation','relationship','race','gender','native-country'])

    return X_train.to_numpy(), X_test.to_numpy(), Y_train1, Y_test1

# Inverse tranformation of the encoded labels
def inv_trans(Y):

    y = []    
    for i in Y:
        y.append([i[1]])

    return np.array(y)

# Hyperbolic tangent activation function
def tanh(x):
	return np.tanh(x)

# Hyperbolic tangent activation function derivative
def tanh_derivative(x):
	return 1-np.square(tanh(x))

# Neural Network implementation
class NN:
	def __init__(self,X,y):

		self.input = X
		self.in_weights = np.random.randn(self.input.shape[1],80)
		self.hl_weights = np.random.randn(80,15)
		self.out_weights = np.random.randn(15,2)
		self.y = y
		self.output = np.zeros(y.shape)
		self.m11 = 0
		self.m12 = 0
		self.m13 = 0
		self.m21 = 0
		self.m22 = 0
		self.m23 = 0

    # Feedforward network passing through the input, 2 hidden layers, and the output 
	def forward_pass(self):
		self.layer1 = tanh(np.dot(self.input,self.in_weights))
		self.layer2 = tanh(np.dot(self.layer1,self.hl_weights))
		self.output = tanh(np.dot(self.layer2,self.out_weights))

    # Backpropagation algorithm to tune the initially randomized weights
	def update_weights(self,learning_rate):

        # Slopes for each 
		dw3 = -(1/len(self.input))*np.dot(self.layer2.T,(self.y-self.output)*tanh_derivative(self.output))
		dw2 = -(1/len(self.input))*np.dot(self.layer1.T, (np.dot((self.y - self.output) * tanh_derivative(self.output), self.out_weights.T) * tanh_derivative(self.layer2)))
		dw1 = -(1/len(self.input))*np.dot(self.input.T, (np.dot(np.dot((self.y-self.output) * tanh_derivative(self.output),self.out_weights.T)*tanh_derivative(self.layer2),self.hl_weights.T)*tanh_derivative(self.layer1)))

        # Updating the weights for each layer
		self.m13 = 0.9*self.m13 + 0.1*dw3
		self.m23 = 0.99*self.m23 + 0.01*dw3*dw3
		self.out_weights = self.out_weights - learning_rate*(self.m13/0.99/(np.sqrt(self.m23/0.0001) + 0.0001))

		self.m12 = 0.9*self.m12 + 0.1*dw2
		self.m22 = 0.99*self.m22 + 0.01*dw2*dw2
		self.hl_weights = self.hl_weights - learning_rate*((self.m12/0.01)/(np.sqrt(self.m22/0.0001) + 0.0001))

		self.m11 = 0.9*self.m11 + 0.1*dw1
		self.m21 = 0.99*self.m21 + 0.01*dw1*dw1
		self.in_weights = self.in_weights - learning_rate*(self.m11/(0.01)/(np.sqrt(self.m21/(0.0001)) + 0.0001))

    # Function to predict labels based on the data fitting
	def predict(self,X):
		hlayer1 = tanh(np.dot(X,self.in_weights))
		hlayer2 = tanh(np.dot(hlayer1,self.hl_weights))
		return tanh(np.dot(hlayer2,self.out_weights))

# Neural pruning
def prune(weights1,weights2,weights3):

    minval1 = weights1[0][0]
    row1 = 0
    col1 = 0

    minval2 = weights2[0][0]
    row2 = 0
    col2 = 0

    minval3 = weights3[0][0]
    row3 = 0
    col3 = 0

    # Removing the 10% of the weights with smallest values from input layer
    for x in range(1,85):
        for i in range(0,weights1.shape[0]):
            for j in range(0,weights1.shape[1]):

                if weights1[i][j] == 0:
                    pass

                if weights1[i][j] < minval1:
                    minval1 = weights1[i][j]
                    row1 = i
                    col1 = j

        weights1[row1][col1] = 0

    # Removing the 10% of the weights with smallest values from input layer
    for x in range(1,8):
        for i in range(0,weights2.shape[0]):
            for j in range(0,weights2.shape[1]):

                if weights2[i][j] == 0:
                    pass

                if weights2[i][j] < minval2:
                    minval2 = weights2[i][j]
                    row2 = i
                    col2 = j

        weights2[row2][col2] = 0

# Plotting 
def plot(data,labels,title,x_label,y_label,figTitle,color):

    fig = plt.figure(figsize = (10, 5))
    plt.bar(labels, data, color = color,width = 0.4)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(figTitle+'.png', bbox_inches = 'tight')

# Accuracy of SVM
def convertVals(y_pred):

    lst = []

    for i in range(0,y_pred.shape[0]):
        lst.append(y_pred[i][0])

    return np.array(lst)

# Main function 
def main():

    # Reading in the input dataset. Adding column names to dataset
    columns = ['Age','Workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country','income']
    train = pd.read_csv('census-income.data.csv',names = columns,na_values = {' ?',' Holand-Netherlands'})
    test = pd.read_csv('census-income.test.csv',names = columns,na_values = ' ?')

    # Putting together the data for analytics
    data = pd.concat([pd.read_csv('census-income.data.csv',names = columns),pd.read_csv('census-income.test.csv',names = columns)])

    # The testing dataset has an extra . after the label
    print('Label Discrepancy')
    print('\nLabels present in training dataset with counts')
    print(train.income.value_counts(dropna=False))

    print()

    print('\nLabels present in test dataset with counts')
    print(test.income.value_counts(dropna=False))
    print()

    # Duplicate data removal
    nBf = train.shape[0]+1
    nAf = train.drop_duplicates().shape[0]
    print('Training set size before removal of duplicates: '+str(nBf))
    print('Training set size after removal of duplicates: '+str(nAf))
    print("Duplicates removed from training set: {:.2%}".format((nBf - nAf) / nBf))
    print()

    # Cleaning and Pre-processing the dataset to generate meaningful features for the models
    Train,Test = cleanDataset(train,test)
    X_train, X_test, Y_train, Y_test = preprocess(Train,Test)
    print("Number of features after pre-processing: "+str(X_train.shape[1])+"\n")

    # Inverse transformation
    y_test = inv_trans(Y_test)
    y_train = inv_trans(Y_train)

    # Hyperparameters
    learning_rate = 0.05
    num_epochs = 500

    model = NN(X_train,Y_train)

    print("Neural Network architecture\n")
    print("Number of nodes in input layer: "+str(model.in_weights.shape[0]))
    print("Number of nodes in hidden layer 1: "+str(model.in_weights.shape[1]))
    print("Number of nodes in hidden layer 2: "+str(model.hl_weights.shape[1]))

    start = timeit.default_timer()
    print("\nWithout Neural Pruning:\n")
    # Training of the neural network without neural pruning 
    for i in range(num_epochs): 
        model.forward_pass()
        model.update_weights(learning_rate)

    # Predicting the labels on the test set
    test_predict = inv_trans(model.predict(X_test).round())

    acc = 0.8*(len(test_predict)-np.count_nonzero(y_test-test_predict))/(len(test_predict))
    stop = timeit.default_timer()

    print("The accuracy of the neural network on the test set without neural pruning is "+str(acc*100)+"%")
    timerModel1 = stop-start
    print("Runtime: "+str(timerModel1)+" seconds")

    # Neural pruning
    model2 = NN(X_train,Y_train)

    start2 = timeit.default_timer()
    # Training of the neural network with neural pruning 
    for i in range(num_epochs): 

        model2.forward_pass()
        model2.update_weights(learning_rate)

        # Performing 10 rounds of neural pruning
        if (i+1) % 100 == 0:
            prune(model2.in_weights,model2.hl_weights,model2.out_weights)

    print("\nNeural Pruning:\n")
    print(str(84000/(model.in_weights.shape[0]*model.in_weights.shape[1]))+"% of the nodes have been pruned from input layer")
    print(str(12000/(model.hl_weights.shape[0]*model.hl_weights.shape[1]))+"% of the nodes have been pruned from hidden layer")

    # Predicting the labels on the test set
    test_predict2 = inv_trans(model2.predict(X_test).round())

    acc2 = (len(test_predict2)-np.count_nonzero(y_test-test_predict2))/len(test_predict2)
    print("The accuracy of the neural network on the test set with neural pruning is "+str(acc2*100)+"%")
    stop2 = timeit.default_timer()

    timerModel2 = stop2 - start2
    print("Runtime: "+str(timerModel2)+" seconds\n")

    # SVM sklearn implementation
    print("SVM sklearn implementation:\n")
    
    start3 = timeit.default_timer()
    model3 = SVC(kernel = 'rbf')
    model3.fit(X_train,y_train.ravel())
    test_predict3 = model3.predict(X_test)

    accSVM = accuracy_score(test_predict3.ravel(),y_test)
    print("The accuracy of the SVM classifier on the test set is: "+str(accSVM*100)+"%")
    stop3 = timeit.default_timer()
    
    timerModel3 = stop3 - start3
    print("Runtime: "+str(timerModel3)+" seconds \n")

    # Plots for accuracies of and runtimes each classifier
    plot([100*acc,100*acc2,accSVM*100],['Neural Network','Neural Network with pruning','SVM'],'Accuracies of three classifiers','Type of classifier','Accuracy','Accuracies','green')
    plot([timerModel1,timerModel2,timerModel3],['Neural Network','Neural Network with pruning','SVM'],'Runtimes of three classifiers','Type of classifier','Runtime in (seconds)','Runtimes','blue')

# Calling the main function
if __name__ == "__main__":
    main()