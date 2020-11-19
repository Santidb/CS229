"""
File: folder_organizer.py
--------------------
This program takes a directory of scraped csv files, containing information for moneyline, spread and
over_under bets, and organizes those files into three separate folders for each type of bet
"""

import os
import shutil

def move_files():
    """ Take a directory of files where files that end with:
            0 belong to moneyline bets
            1 belong to spread bets
            2 belong to over_under bets

    And distribute those files among three folders, categorizing each file into respective bet folder
    """

    # Change working directory
    os.chdir("NBA 2020")

    # Obtain all files in current working directory
    for file in os.listdir():

        # If file ends with .csv (ignore other files)
        if file[-4:] == ".csv":

            # Obtaining filename without the .csv
            filename = file[:-4]

            # Conditions on last number determine type of bet
            if filename[-1] == "0":
                shutil.move(file, "Moneyline")
            elif filename[-1] == "1":
                shutil.move(file, "Spread")
            elif filename[-1] == "2":
                shutil.move(file, "Over_Under")

def main():
    move_files()

if __name__ == "__main__":
    main()