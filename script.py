import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]

    
    data_with_bias = np.column_stack((np.ones(n_data), train_data))
    
    
    z = np.dot(data_with_bias, initialWeights)
    
    
    theta = sigmoid(z)

    
    labeli = labeli.reshape(theta.shape)  

    error = -(1/n_data) * np.sum(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta))
    
    error_grad = (1/n_data) * np.dot(data_with_bias.T, (theta - labeli))

    return error, error_grad.flatten()

def blrPredict(W, data):
    """
    blrPredict predicts the label of data given the data and parameter W 
    of Logistic Regression
    """
    
    data_with_bias = np.column_stack((np.ones(data.shape[0]), data))
    
    posterior_probs = sigmoid(np.dot(data_with_bias, W))
    
    label = np.argmax(posterior_probs, axis=1).reshape(-1, 1)

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_class = 10
    
    W = params.reshape((n_feature + 1, n_class))
    
    data_with_bias = np.column_stack((np.ones(n_data), train_data))
    
    z = np.dot(data_with_bias, W)
    
    exp_z = np.exp(z)
    softmax_prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    error = -np.sum(labeli * np.log(softmax_prob)) / n_data
    
    error_grad = np.dot(data_with_bias.T, (softmax_prob - labeli)) / n_data
    
    return error, error_grad.flatten()

def mlrPredict(W, data):
    """
    mlrPredict predicts the label of data given the data and parameter W
    of multi-class Logistic Regression
    """
    data_with_bias = np.column_stack((np.ones(data.shape[0]), data))
    
    z = np.dot(data_with_bias, W)
    
    exp_z = np.exp(z)
    softmax_prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    label = np.argmax(softmax_prob, axis=1).reshape(-1, 1)

    return label

if __name__ == "__main__":
    """
    Script for Logistic Regression
    """
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
    
    # number of classes
    n_class = 10
    
    # number of training samples
    n_train = train_data.shape[0]
    
    # number of features
    n_feature = train_data.shape[1]
    
    Y = np.zeros((n_train, n_class))
    for i in range(n_class):
        Y[:, i] = (train_label == i).astype(int).ravel()
    
    # Logistic Regression with Gradient Descent
    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros(n_feature + 1)
    opts = {'maxiter': 100}
    for i in range(n_class):
      labeli = Y[:, i].reshape(n_train, 1)
      args = (train_data, labeli)
      nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
      W[:, i] = nn_params.x.reshape((n_feature + 1,))
    
    predicted_label = blrPredict(W, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
    
    predicted_label = blrPredict(W, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
    
    predicted_label = blrPredict(W, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
    
    """
    Script for Support Vector Machine
    """
    print('\n\n--------------SVM-------------------\n\n')

    np.random.seed(42)
    smpl_indices = np.random.choice(train_data.shape[0], 10000, replace=False)
    smpl_trn_data = train_data[smpl_indices]
    sample_train_label = train_label[smpl_indices]

    print("Linear Kernel SVM:")
    clf_lin = svm.SVC(kernel='linear')
    clf_lin.fit(smpl_trn_data, sample_train_label.ravel())

    print('Training Accuracy:', clf_lin.score(smpl_trn_data, sample_train_label.ravel()) * 100, '%')
    print('Validation Accuracy:', clf_lin.score(validation_data, validation_label.ravel()) * 100, '%')
    print('Testing Accuracy:', clf_lin.score(test_data, test_label.ravel()) * 100, '%')

    print("\nRBF Kernel SVM (gamma=1):")
    clf_rbf_1 = svm.SVC(kernel='rbf', gamma=1)
    clf_rbf_1.fit(smpl_trn_data, sample_train_label.ravel())

    print('Training Accuracy:', clf_rbf_1.score(smpl_trn_data, sample_train_label.ravel()) * 100, '%')
    print('Validation Accuracy:', clf_rbf_1.score(validation_data, validation_label.ravel()) * 100, '%')
    print('Testing Accuracy:', clf_rbf_1.score(test_data, test_label.ravel()) * 100, '%')

    print("\nRBF Kernel SVM (default gamma):")
    clf_rbf_default = svm.SVC(kernel='rbf')
    clf_rbf_default.fit(smpl_trn_data, sample_train_label.ravel())

    print('Training Accuracy:', clf_rbf_default.score(smpl_trn_data, sample_train_label.ravel()) * 100, '%')
    print('Validation Accuracy:', clf_rbf_default.score(validation_data, validation_label.ravel()) * 100, '%')
    print('Testing Accuracy:', clf_rbf_default.score(test_data, test_label.ravel()) * 100, '%')

    print("\nRBF Kernel SVM with varying C:")
    C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracies = []

    for C in C_values:
        clf = svm.SVC(kernel='rbf', C=C)
        clf.fit(smpl_trn_data, sample_train_label.ravel())
        accuracies.append(clf.score(validation_data, validation_label.ravel()))

    plt.figure(figsize=(10, 6))
    plt.plot(C_values, accuracies, marker='o')
    plt.title('SVM Accuracy vs Regularization Parameter C')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.show()

    best_C = C_values[np.argmax(accuracies)]
    print(f"\nBest C value: {best_C}")

    clf_best = svm.SVC(kernel='rbf', C=best_C)
    clf_best.fit(train_data, train_label.ravel())

    print('Training Accuracy:', clf_best.score(train_data, train_label.ravel()) * 100, '%')
    print('Validation Accuracy:', clf_best.score(validation_data, validation_label.ravel()) * 100, '%')
    print('Testing Accuracy:', clf_best.score(test_data, test_label.ravel()) * 100, '%')
    
    """
    Script for Extra Credit Part
    """
    # FOR EXTRA CREDIT ONLY
    W_b = np.zeros((n_feature + 1, n_class))
    initialWeights_b = np.zeros((n_feature + 1, n_class))
    opts_b = {'maxiter': 100}
    
    args_b = (train_data, Y)
    nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))
    
    predicted_label_b = mlrPredict(W_b, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
    
    predicted_label_b = mlrPredict(W_b, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
    
    predicted_label_b = mlrPredict(W_b, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')