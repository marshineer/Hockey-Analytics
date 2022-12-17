import os
import csv
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
sum_dist = 0
n_shots = 0
n_events = 0
t_start = time()
print('Starting coordinate corrections')
for i, event_x in enumerate(all_events):
    # Fix assists on empty net goals
    if event_x['emptyNet']:
        event_x['assist2ID'] = event_x.get('assist1ID', None)
        event_x['assist1ID'] = event_x.get('player2ID', None)

    # Flip coordinates of regular season OT prior to the 2014-15 season
    event_x['xCoord'] = float(event_x['xCoord'])
    event_x['yCoord'] = float(event_x['yCoord'])
    game_id = event_x['GameID']
    if game_id[:4] > '2013' and game_id[4:6] == '02':
        event_x['xCoord'] = -event_x['xCoord']
        event_x['yCoord'] = -event_x['yCoord']

    # Calculate the shot distances from the net
    if event_x['eventTypeId'] in shot_types:
        x = event_x['xCoord']
        y = event_x['yCoord']
        period = int(event_x['period'])
        home_shot = eval(event_x['player1Home'])
        if event_x['eventTypeId'] == 'BLOCK':
            home_shot = not home_shot
        home_end = bool(home_shot) ^ bool(period % 2 == 1)
        this_dist = calc_coord_diff(x, y, home_end=home_end)
        sum_dist += this_dist
        n_shots += 1

    # Correct the coordinates for all events
    n_events += 1
    if (game_id != last_game_id) or (i == len(all_events) - 1):
        avg_shot_dist = sum_dist / n_shots
        if avg_shot_dist > 89:
            if i == len(all_events) - 1:
                for event_y in all_events[i - n_events:]:
                    if event_y['eventTypeId'] in shot_types:
                        event_y['xCoord'] = -event_y['xCoord']
                        event_y['yCoord'] = -event_y['yCoord']
            else:
                for event_y in all_events[i - n_events:i]:
                    if event_y['eventTypeId'] in shot_types:
                        event_y['xCoord'] = -event_y['xCoord']
                        event_y['yCoord'] = -event_y['yCoord']
        last_game_id = game_id
        sum_dist = 0
        n_shots = 0
        n_events = 0
print(f'It took {timedelta(seconds=(time() - t_start))} to fix the coordinates')

# Save the new shot table and updated event table
save_nhl_data(froot + '/../data/game_events.csv', all_events, overwrite=True)
