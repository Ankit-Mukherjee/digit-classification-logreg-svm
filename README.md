# Handwritten Digit Classification with Logistic Regression and SVM

 This project implements handwritten digit classification using Logistic Regression and Support Vector Machines (SVM) on the MNIST dataset.
 Additionally, a multi-class logistic regression model is implemented for extra credit.

 -----------------------------------------------------------------------------

# 1. Prerequisites

 Python 3.x
 Required Libraries:
 - `numpy`
 - `scipy`
 - `matplotlib`
 - `scikit-learn`

# Install the libraries using:
```bash
pip install numpy scipy matplotlib scikit-learn
```
 -----------------------------------------------------------------------------

# 2. Dataset

 The project uses the MNIST dataset in the form of the `mnist_all.mat` file.
 Ensure the file is placed in the project directory before running the script.

 -----------------------------------------------------------------------------

# 3. Project Structure

 - `preprocess()`: Loads the MNIST dataset, normalizes it, and splits it into training, validation, and test sets.
 - `blrObjFunction()`: Implements the error function and gradient calculation for binary logistic regression.
 - `blrPredict()`: Predicts class labels using trained logistic regression classifiers.
 - `mlrObjFunction()`: Computes the error function and gradient for multi-class logistic regression.
 - `mlrPredict()`: Predicts class labels using a trained multi-class logistic regression model.
 - `SVM Implementation`: Includes training and evaluation of SVM classifiers with different kernels and hyperparameter settings.

 -----------------------------------------------------------------------------

# 4. Usage

 To run the project, execute the following command:
 ```bash
python script.py
```

# Outputs:
 1. Training, validation, and test accuracies for:
    - Logistic Regression (One-vs-All)
    - Multi-class Logistic Regression
    - SVM with different kernel functions and parameters.
 2. Visualization:
    - A plot showing accuracy trends for SVM with RBF kernel as the regularization parameter `C` varies.

 -----------------------------------------------------------------------------

# 5. Features

 Logistic Regression:
 - Implements one-vs-all (OvA) classification for multi-class problems.
 - Functions:
   - Training: blrObjFunction()
   - Prediction: blrPredict()

 Multi-Class Logistic Regression:
 - Implements a single logistic regression model with a softmax layer.
 - Functions:
   - Training: mlrObjFunction()
   - Prediction: mlrPredict()

 Support Vector Machines:
 - SVM experiments with the following configurations:
   - Linear kernel
   - RBF kernel with:
     - gamma = 1
     - Default gamma (1/number of features)
     - Hyperparameter tuning over varying C values.
 - Best-performing parameters are identified, and the model is trained on the entire dataset.

 -----------------------------------------------------------------------------

# 6. Results

 - Outputs include the accuracy for training, validation, and test sets for all classifiers.
 - Identifies the optimal value of C for SVM with RBF kernel.
 - Displays the impact of kernel selection and parameter tuning on classification performance.

 -----------------------------------------------------------------------------


# 8. Visualization

 The project generates a plot showing the accuracy trend for SVM with RBF kernel across varying values of the regularization parameter C.
