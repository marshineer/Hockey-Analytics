import os
import csv
import pandas as pd
from time import time
from datetime import timedelta
from nhl_api.ref_common import calc_coord_diff
from nhl_api.common import save_nhl_data

# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the event data as a list
with open(froot + '/../data/game_events.csv', 'r') as f:
    dict_reader = csv.DictReader(f)
    all_events = list(dict_reader)

# Fix the coordinate direction when flipped
shot_types = ['GOAL', 'SHOT', 'MISS', 'BLOCK']
last_game_id = all_events[0]['GameID']
ind0 = 0
fix_game_ids = []
t_start = time()
print('Starting net distance calculations')
for i, event_x in enumerate(all_events):
    # Fix assists on empty net goals
    if event_x['emptyNet']:
        event_x['assist2ID'] = event_x.get('assist1ID', None)
        event_x['assist1ID'] = event_x.get('player2ID', None)

    event_x['xCoord'] = float(event_x['xCoord'])
    event_x['yCoord'] = float(event_x['yCoord'])
    if event_x['eventTypeId'] in shot_types:
        x = event_x['xCoord']
        y = event_x['yCoord']
        period = int(event_x['period'])
        home_shot = eval(event_x['player1Home'])
        if event_x['eventTypeId'] == 'BLOCK':
            home_shot = not home_shot
        home_end = bool(home_shot) ^ bool(period % 2 == 1)
        event_x['netDistance'] = calc_coord_diff(x, y, home_end=home_end)
    else:
        event_x['netDistance'] = None
    game_id = event_x['GameID']
    if game_id != last_game_id:
        game_df = pd.DataFrame(all_events[ind0:i])
        game_shots = game_df.loc[game_df.eventTypeId.isin(shot_types)]
        avg_shot_dist = sum(game_shots.netDistance) / len(game_shots)
        if avg_shot_dist > 95:
            fix_game_ids.append(last_game_id)
        last_game_id = game_id
        ind0 = i
    elif i == len(all_events) - 1:
        game_df = pd.DataFrame(all_events[ind0:])
        game_shots = game_df.loc[game_df.eventTypeId.isin(shot_types)]
        avg_shot_dist = sum(game_shots.netDistance) / len(game_shots)
        if avg_shot_dist > 95:
            fix_game_ids.append(last_game_id)
print(f'It took {timedelta(seconds=(time() - t_start))} to calculate distances')

t_start = time()
print('Starting coordinate corrections')
events_df = pd.DataFrame(all_events)
for event_x in all_events:
    if event_x['GameID'] in fix_game_ids:
        event_x['xCoord'] = -event_x['xCoord']
        event_x['yCoord'] = -event_x['yCoord']
print(f'It took {timedelta(seconds=(time() - t_start))} to fix the coordinates')

# Save the new shot table and updated event table
events_df.drop(columns='netDistance', inplace=True)
event_list = events_df.to_dict('records')
save_nhl_data(froot + '/../data/game_events.csv', event_list, overwrite=True)
