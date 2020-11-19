"""
File: custom_loss.py
--------------------
This program creates a custom loss function that can be used to train the logistic regression model for
the sport betting case
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from util import load_dataset

class LogisticRegression:
    """Logistic regression with Gradient Descent as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=False, debug=False):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.loss = None
        self.debug = debug

        self.iteration_profit = []
        self.theta_history = []
    
    def fit(self, x, y, odds_home, odds_away):
        """Run Gradient Descent to maximize bet profitability for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # Capture number of observations
        num_obs = x.shape[0]
        dim = x.shape[1]

        # Preparing data
        y = y.flatten()

        # If theta_0 is None, then initialize with the vector of 0
        if self.theta == None:
            self.theta = np.zeros((dim))

        # Run gradient descent to optimize theta
        # Initialize the norm at 1, loss and iteration counter
        norm = 1
        counter = 1
        y_hat = self.predict(x)
        self.loss = custom_loss(y, y_hat, odds_home, odds_away)

        # Begin loop
        while norm > self.eps and counter <= self.max_iter:
        # while counter < 20:

            # Running an update in theta
            theta = update_theta(self.theta, x, y, odds_home, odds_away, self.step_size)

            # Calculating norm
            norm = np.sum(np.abs(theta - self.theta))

            # Storing the new theta
            self.theta = theta

            # Make predictions at this point
            y_hat = self.predict(x)

            print(f"Number of games predicted win home: {y_hat.sum()}")
            print(f"Total number of games: {len(y_hat)}")

            # Calculating loss with this theta
            loss_new = custom_loss(y, y_hat, odds_home, odds_away)

            # Computing difference between loss functions
            diff_loss = np.abs(loss_new - self.loss)

            # Updating loss
            self.loss = loss_new

            # Debugging prints
            if self.verbose:
              print("-"*40)
              print(f"Iteration {counter}")
              print(f"Norm is {norm}")
              print(f"Profit is {self.loss}")
              print(f"Difference is {diff_loss}")

            # Saving model profit
            if self.debug:
              self.iteration_profit.append(self.loss)
              self.theta_history.append(self.theta)


            # Updating iteration
            counter += 1

    def predict(self, x):
        """Return decision with highest probability given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        linear_reg = x.dot(self.theta)
        y_prob = expit(linear_reg)

        # Create matrix with y=1 if prob > 0.5
        y_hat = np.where(y_prob > 0.5, 1, 0)

        return y_hat

    def predict_proba(self, x):
      """ Return predicted soft probabilities given new inputs x

      Args:
        x: Inputs of shape (n_examples, dim)

      Returns:
        Outputs of shape (n_examples,) 
      """
      linear_reg = x.dot(self.theta)
      y_prob = expit(linear_reg)

      return y_prob


    def predict_debug(self, x, y, odds_home, odds_away):
        """ Returns profit at every iteration. Useful for debugging optimization process

        Returns:
            Output of shape (num_iterations,)
        """
        acc_history = []
        for i in range(len(self.theta_history)):
          linear_reg = x.dot(self.theta_history[i])
          y_prob = expit(linear_reg)
          y_hat = np.where(y_prob > 0.5, 1, 0)
          profit = custom_loss(y, y_hat, odds_home, odds_away)
          acc_history.append(profit)

        return acc_history




def custom_loss(true_label, predict_label, odds_home, odds_away):
    """ Computes total betting loss given a series of bets to be made and odds at each time point

    Args:
        true_label: outcomes of the match, 0 is for home team loses, 1 is for home team wins, size (n,)
        predict_label: predictions of the model, size (n,)
        odds_home: odds that bookies pay for a win at home, size (n,)
        odds_away: odds that bookies pay for a win away, size (n,)

    Returns:
        loss: Total betting loss, scalar
    """
    loss_vec = true_label * (predict_label * (odds_home - 1) + predict_label - 1) + \
            (1-true_label) * ((1-predict_label)*(odds_away - 1) - predict_label)
    return loss_vec.sum()


def sigmoid(z):
    """ Computes sigmoid function """
    # sig = 1 / (1 + np.exp(-np.clip(z, 1e-12, None)))
    sig = 1 / (1 + np.exp(-z))
    print("hello")
    return sig


def update_theta(theta, x, y, odds_home, odds_away, lr):
    """Problem: given the current value of theta and the learning rate lr,
    you should return the new value of theta obtained by running 1 iteration
    of the gradient descend algorithm.

    Args:
        theta: the current theta
        lr: the learning rate

    Returns:
        the new value of theta after 1 iteration of gradient descend
    """
    # Lets first calculate the derivative of the loss function
    n = x.shape[0]
    dim = x.shape[1]

    # Now we are ready to calculate the gradient
    z = x.dot(theta)

    # Calculating gradient
    sig_matrix = (1 - expit(z)) * expit(z)
    odds_matrix = y * odds_home - (1-y) * odds_away
    # Combining everything that doesn't depend on x, converting into array of appropriate size
    all_matrix = sig_matrix * odds_matrix
    all_matrix = all_matrix.reshape((n,1))
    # Adding the matrix, this will yield a (N,dim) matrix that we can then collapse
    full_grad = all_matrix * x

    # Calculating gradient along each theta
    gradient = full_grad.sum(axis=0)

    # Now we can calculate the gradient ascent rule
    theta_new = theta + lr * gradient

    return theta_new


def main():
    # Read data
    x_train = load_dataset("x_train.csv", intercept=True)
    y_train = load_dataset("y_train.csv").to_numpy()
    x_val = load_dataset("x_val.csv", intercept=True)
    y_val = load_dataset("y_val.csv").to_numpy()

    odds_home = x_train['Odds_Home'].to_numpy()
    odds_away = x_train['Odds_Away'].to_numpy()
    odds_home_val = x_val['Odds_Home'].to_numpy()
    odds_away_val = x_val['Odds_Away'].to_numpy()
    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()

    # Normalize variables
    # all_vars = x_train.columns.values.tolist()
    # norm_vars = ['Time_left', 'Spread', 'Odds_Home', 'Odds_Away', 'Initial_odds_home', 'Initial_odds_away']
    # other_vars = [i for i in all_vars if i not in norm_vars]
    # norm_df = normalize(x_train[norm_vars], axis=0)
    # other_df = x_train[other_vars].to_numpy()
    # x_train = np.concatenate((norm_df, other_df), axis=1)

    # Define a logistic regression model
    # lr_vec = [1e-4, 1e-5, 8e-6, 5e-6, 3e-6]
    lr_vec = [8e-6]
    max_profit_train = []
    max_profit_val = []
    models = []

    for lr in lr_vec:
        # Initializing and training the model
        model = LogisticRegression(step_size=lr, verbose=True)
        model.fit(x_train, y_train, odds_home, odds_away)

        # Generating predictions and calculating profit
        y_val_hat = model.predict(x_val)
        y_val_profit = custom_loss(y_val.flatten(), y_val_hat, odds_home_val, odds_away_val)

        # Appending profit in training and validation
        max_profit_train.append(model.loss)
        max_profit_val.append(y_val_profit)
        models.append(model)

    print("-"*40)
    print(lr_vec)
    print(max_profit_train)
    print(max_profit_val)

    # print(y_train_hat)
    # pd.DataFrame(y_train_hat).to_csv("y_train_hat_bets.csv")


    # Run the logistic regression model

    # Predict labels

    # Compute betting loss

if __name__ == "__main__":
    main()