import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count += 1
        loss += margin
        dW[:, j] += X[i]
    dW[:, y[i]] -= count * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W 

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  mask = np.zeros((X.shape[0], W.shape[1]))
  XW = np.matmul(X, W)
  labels = XW[range(XW.shape[0]), y].reshape(y.shape[0], 1)
  loss = np.sum(np.maximum(XW - labels + 1, mask)) - X.shape[0]
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  count = np.maximum(XW - labels + 1, mask)
  count[range(XW.shape[0]), y] -= 1
  count = (count > 0)

  y_new = np.zeros((X.shape[0], W.shape[1]))
  y_new[range(y_new.shape[0]), y] = 1
  y_new = y_new * np.sum(count, axis=1).reshape(count.shape[0], 1)
  y_new = y_new.reshape(X.shape[0], 1, W.shape[1])
  X_new = np.repeat(X, 10).reshape(X.shape[0], X.shape[1], W.shape[1])
  prod = np.sum(X_new * y_new, axis=0).reshape(X.shape[1], W.shape[1])
  dW -= prod
  prod_new = np.sum(X_new * count.reshape(X.shape[0], 1, W.shape[1]), axis=0).reshape(X.shape[1], W.shape[1])
  dW += prod_new
  
  dW /= X.shape[0]
  dW += 2 * reg * W 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
