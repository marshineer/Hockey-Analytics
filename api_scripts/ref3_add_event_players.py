import os
import csv
import pandas as pd
from time import time
from datetime import timedelta
from nhl_api.ref_common import add_players_to_events
from nhl_api.common import save_nhl_data


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the game info
games_df = pd.read_csv(froot + f'/../data/games.csv')
games_dict = games_df.to_dict('records')
games = {game_x['GameID']: game_x for game_x in games_dict}

# Convert strings of active players back to lists
for g_dict in games.values():
    g_dict['activeHomePlayers'] = eval(g_dict['activeHomePlayers'])
    g_dict['activeAwayPlayers'] = eval(g_dict['activeAwayPlayers'])

# Load the shift data as a data frame and event data as a list
shifts_df = pd.read_csv(froot + f'/../data/shifts.csv')
with open(froot + f'/../data/game_events.csv', 'r') as f:
    dict_reader = csv.DictReader(f)
    all_events = list(dict_reader)

# Add players on ice to the game event data
t_start = time()
print('Starting player updates using tables')
add_players_to_events(all_events, shifts_df, games)
print(f'It took {timedelta(seconds=(time() - t_start))} to add players to '
      f'the events of {len(games)} games ({len(all_events)} events)')

# Save the updated game event data
save_nhl_data(froot + f'/../data/game_events.csv', all_events, overwrite=True)
