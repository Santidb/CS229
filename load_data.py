"""
File: load_data.py
--------------------
This is a program to take individual csv files, modify them to get the appropriate header names and variables,
and save all of the matches in a single database. Final database is saved as a .pkl file.
"""

import math
import numpy as np
import pandas as pd
import os

pd.options.display.width = 0
# pd.options.display.max_rows = 100

def mod_csv(filename, bet, match_id):
    """ Takes in a file from scraper, cleans it up by (1) renaming columns, (2) calculating relevant variables such as
    score spread, time left, quarter, and (3) ordering everything in a pandas dataframe.

    :param filename: filename of csv file to modify
    :param bet: type of bet - moneyline, over_under or spread

    :return: pandas dataframe with clean data for this individual file
    """

    # Reading data. Note that chr(92) = \.
    folder = f'NBA 2020{chr(92)}'
    filepath = folder + bet + chr(92) + filename
    dataframe = pd.read_csv(filepath)

    # Transforming data
    if bet == "Moneyline":
        cols = dataframe.columns.values

        # Obtaining home team name and score
        title_home = cols[1]
        team_home = title_home[:title_home.find("-")-1].strip()
        score_home = title_home[title_home.find("-")+2:].strip()
        score_home = int(score_home)

        # Obtaining away team name and score
        title_away = cols[2]
        team_away = title_away[:title_away.find("-") - 1].strip()
        score_away = title_away[title_away.find("-") + 2:].strip()
        score_away = int(score_away)

        # Obtaining match date and stadium
        match_date = cols[3].strip()
        stadium = cols[4].strip()

        # Renaming columns
        # dataframe.columns = ["Match_time", "Score", "Odds_Home", "Odds_Away", "Timestamp"]

        dataframe.rename(columns={cols[0]: "Match_time", cols[1]: "Score", cols[2]: "Odds_Home",
                          cols[3]: "Odds_Away", cols[4]: "Timestamp"}, inplace=True)

        # Parsing out score for each team
        dataframe['dash_pos'] = dataframe['Score'].str.find("-")
        dataframe['Score_Home'] = dataframe.apply(lambda x: x['Score'][0:x['dash_pos']], axis=1).astype('int64')
        dataframe['Score_Away'] = dataframe.apply(lambda x: x['Score'][x['dash_pos']+1:], axis=1).astype('int64')

        # Eliminating Score, dash_pos (irrelevant variables)
        dataframe.drop(columns=['dash_pos', 'Score'], inplace=True)
        # Creating spread variable
        dataframe['Spread'] = dataframe['Score_Home'] - dataframe['Score_Away']

        ############# Creating "Time Left on Clock" variable ###################

        # Extracting quarter played
        dataframe['quarter'] = dataframe['Match_time'].str[:2]

        # Saving quarter in Q
        dataframe.loc[dataframe['quarter'] == "Q4", 'Q'] = 4
        dataframe.loc[dataframe['quarter'] == "Q3", 'Q'] = 3
        dataframe.loc[dataframe['quarter'] == "Q2", 'Q'] = 2
        dataframe.loc[dataframe['quarter'] == "Q1", 'Q'] = 1
        dataframe.loc[dataframe['quarter'].str[:1] != "Q", 'Q'] = 1
        dataframe.loc[dataframe['quarter'].str[:1] != "Q", 'Match_time'] = "Pre-match"

        # Minutes left in quarter
        # Finding position of : in match_time
        dataframe['dash_pos'] = dataframe['Match_time'].str.find(":")
        # Extracting minutes from match_time
        dataframe['minutes'] = dataframe.apply(lambda x: x['Match_time'][x['dash_pos']-2:x['dash_pos']], axis=1)
        # if no : present in match_time, 12 minutes left in the quarter
        dataframe.loc[dataframe['dash_pos'] == -1, 'minutes'] = 12
        # If Match_time ends with "Ended", then 0 minutes left in the quarter
        dataframe.loc[dataframe['Match_time'].str[-5:] == "Ended", 'minutes'] = 0
        dataframe['minutes'] = dataframe['minutes'].astype('int64')

        # Seconds left in minute
        dataframe['seconds'] = dataframe.apply(lambda x: x['Match_time'][x['dash_pos']+1:], axis=1)
        dataframe.loc[dataframe['dash_pos'] == -1, 'seconds'] = 0
        dataframe['seconds'] = dataframe['seconds'].astype('int64')

        # Calculating seconds left in the game
        dataframe['Time_left'] = 12*(4-dataframe['Q']) + dataframe['minutes'] + dataframe['seconds']/60

        # Dropping irrelevant columns
        dataframe.drop(columns=['quarter', 'dash_pos', 'minutes', 'seconds'], inplace=True)

        # Adding teams to columns
        dataframe['Team_Home'] = team_home
        dataframe['Team_Away'] = team_away
        dataframe['Match_date'] = match_date
        dataframe['ID'] = match_id
        dataframe['Winner'] = dataframe.apply(lambda x: 1 if score_home > score_away else 0, axis=1)
        dataframe['Initial_odds_home'] = dataframe['Odds_Home'][0]
        dataframe['Initial_odds_away'] = dataframe['Odds_Away'][0]

        # Rearranging columns
        cols = dataframe.columns.values

        new_df = dataframe[[cols[12], cols[11], cols[3], cols[0], cols[7], cols[8], cols[9], cols[10],
                            cols[4], cols[5], cols[6], cols[1], cols[2], cols[14], cols[15], cols[13]]]

        return new_df


def obtain_filelist():
    """ Obtain a list with all of the files corresponding to the moneyline bet data

    :return: list with all csv files
    """
    filelist = []

    for file in os.listdir("NBA 2020\Moneyline"):

        # If file ends with .csv (ignore other files)
        if file[-4:] == ".csv":

            filelist.append(file)

    return filelist


def gen_matchid(counter):
    """ Generates a string containing match id for the counter

    :param counter: counter of iterations
    :return: match_id key
    """
    if len(str(counter)) == 1:
        match_id = "00000" + str(counter)
    elif len(str(counter)) == 2:
        match_id = "0000" + str(counter)
    elif len(str(counter)) == 3:
        match_id = "000" + str(counter)
    elif len(str(counter)) == 4:
        match_id = "00" + str(counter)
    elif len(str(counter)) == 5:
        match_id = "0" + str(counter)
    else:
        match_id = str(counter)

    return match_id


def main():
    """ Obtains filelist, then modifies every csv and appends into a list with all matches. All matches are
    concatenated into a single dataframe.
    """

    filelist = obtain_filelist()
    all_df = []

    for counter in np.arange(0, len(filelist)):

        # Generating match id
        match_id = gen_matchid(counter)

        # Modifying each csv
        print(filelist[counter])
        print(counter)

        df = mod_csv(filelist[counter], bet="Moneyline", match_id=match_id)

        # Appending into a list with all dataframes from all matches
        all_df.append(df)
        counter += 1

    df_total = pd.concat(all_df, ignore_index=True)

    # Saving dataset
    df_total.to_pickle("Moneyline_alldata.pkl")

if __name__ == "__main__":
    main()

