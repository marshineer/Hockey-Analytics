import os
import csv
from nhl_api.common import save_nhl_data


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the game data as a list
with open(froot + '/../data/games.csv', 'r') as f:
    dict_reader = csv.DictReader(f)
    all_games = list(dict_reader)

for game in all_games:
    game['home_win'] = game['homeScore'] > game['awayScore']

# Save the updated games table
save_nhl_data(froot + '/../data/games.csv', all_games, overwrite=True)
