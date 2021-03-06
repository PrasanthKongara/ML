"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    d=(np.dot(X,w.T)-np.array(y))
    rss=np.sum(d*d)
    err=np.divide(rss,len(y),dtype="float64")
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  cov=np.dot(X.T,X)
  cov_inv=np.linalg.inv(cov)
  m=np.dot(cov_inv,X.T)
  w=np.dot(m,y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    cov=np.dot(X.T,X)
    e,ev=np.linalg.eig(cov)
    p=np.amin(np.absolute(e))
    while p<10**-5:
        I=np.dot(10**-1,np.identity(len(cov)))
        cov=np.add(cov,I)
        e,ev=np.linalg.eig(cov)
        p=np.amin(np.absolute(e))
    cov_inv=np.linalg.inv(cov)
    m=np.dot(cov_inv,X.T)
    w=np.dot(m,y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################	
    cov=np.dot(X.T,X)
    I=np.dot(lambd,np.identity(len(cov)))
    cov=np.add(cov,I)
    cov_inv=np.linalg.inv(cov)
    m=np.dot(cov_inv,X.T)
    w=np.dot(m,y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    mse=np.zeros(39,dtype="float64")	
    for k in range(-19,20):
        w=regularized_linear_regression(Xtrain, ytrain, 10**k)
        mse[k+19]=mean_square_error(w, Xval, yval)
    i=np.argmin(mse)
    bestlambda=np.array(10,dtype="float64")
    bestlambda=np.power(bestlambda,(i-19))
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    temp=X[:]
    for i in range(2,power+1):
        p=np.power(X,i)
        temp=np.append(temp,p,axis=1)
    X=temp
    return X


