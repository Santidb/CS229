"""
File: util.py
--------------------
Contains utility functions for loading data, and other useful functions we will be using across
our project
"""

import numpy as np
import pandas as pd

def load_dataset(csv_path, intercept=False):
    """ Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of inputs or labels
    """
    data = pd.read_csv(csv_path, index_col=0)

    if intercept:
        data = add_intercept(data)

    return data

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D pandas dataframe.

    Returns:
        New dataframe same as x with 1's in the 0th column.
    """
    x.insert(0, "Intercept", 1)

    return x

def place_bets(predicted_probabilities, odds_home, odds_away):
  """ Given soft predictions from a model and casino payouts, this function
      selects the bet with the highest expected value: bet home, bet away or no bet

  Args:
    predicted_probabilities: soft probabilities from the model, where first column is for home team loss and second column is for home team win
    odds_home: casino payout for home bet
    odds_away: casino payout for away bet

  Returns:
    bet_decisions: vector with values {Bet Home, Bet Away, No Bet}
  """
  # Calculate expected profit for each bet
  bet_home = predicted_probabilities[:,1]*(odds_home-1) - (predicted_probabilities[:,0])
  bet_away = predicted_probabilities[:,0]*(odds_away-1) - (predicted_probabilities[:,1])

  # Obtaining size
  size = len(bet_home)

  # Putting it all together
  bet_matrix = np.zeros((size, 3))
  bet_matrix[:, 0] = bet_home
  bet_matrix[:, 1] = bet_away

  # Choosing bet
  bet_decisions = np.argmax(bet_matrix, axis=1)

  # Creating sparse matrix with binary decisions on each bet
  bets = np.zeros(bet_matrix.shape)
  bets[np.arange(size), bet_decisions] = 1

  return bets

def evaluate_bets(bets, odds_home, odds_away, y):
  """ Given bet decisions, calculate profit of the betting strategy

  Args:
    bets: matrix of size (n,3) with decisions {Bet_home, Bet_away, No_bet}
    odds_home: casino payout for home bet
    odds_away: casino payout for away bet
    y: true results of match

  Returns:
    Profit: Actual profit of following this betting strategy
  """
  profit = bets[:,0] * (y * odds_home - 1) + bets[:,1] * ((1-y)*odds_away - 1)

  return profit












