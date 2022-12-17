import os
import csv
import pandas as pd
from time import time
from datetime import timedelta
from nhl_api.ref_common import game_time_to_sec, calc_coord_diff,\
    calc_net_angle, calc_angle_diff
from nhl_api.common import save_nhl_data


# TODO: Other values that should be incorporated in the shot table:
#  p(goal), p(rebound), p(on_net), p(frozen)
# TODO: Block rebounds should not count towards p(rebound)
# TODO: Update empytNet bool in both events and shots? (No probably not)

# Initialize shot booleans
shot_bools = {'goal': False,
              'missed': False,
              'blocked': False,
              'reboundShot': False,
              'playEnds': False,
              'puckFrozen': False}

# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the event data as a list
with open(froot + '/../data/game_events.csv', 'r') as f:
    dict_reader = csv.DictReader(f)
    all_events = list(dict_reader)

# Fix the coordinate direction when flipped
for i, event_x in enumerate(all_events):
    if event_x['emptyNet']:
        event_x['assist2ID'] = event_x.get('assist1ID', None)
        event_x['assist1ID'] = event_x.get('player2ID', None)
    x = float(event_x['xCoord'])
    y = float(event_x['yCoord'])
    period = int(event_x['period'])
    home_shot = event_x['shooterHome']
    home_end = bool(home_shot) ^ bool(period % 2 == 1)
    event_x['netDistance'] = calc_coord_diff(x, y, home_end=home_end)
    event_x['netAngle'] = calc_net_angle(x, y, home_end=home_end)

shots_df = pd.DataFrame(shot_list)
events_df = pd.DataFrame(all_events)
game_ids = shots_df.GameID.unique().tolist()
t_start = time()
print('Starting coordinate corrections')
for game_id in game_ids:
    game_shots = shots_df.loc[shots_df.GameID == game_id]
    mask_shots = game_shots.index.values
    # mask_shots = shots_df.loc[shots_df.GameID == game_id].index.values
    avg_shot_dist = sum(game_shots.netDistance) / len(game_shots)
    # print(type(game_shots.netDistance.iloc[0]))
    # avg_shot_dist = pd.mean(shots_df.loc[mask_shots, 'netDistance'])
    if avg_shot_dist > 95:
        # print(type(game_shots.xCoord.iloc[0]))
        shots_df.loc[mask_shots, 'xCoord'] = \
            [-float(x) for x in game_shots.xCoord]
        shots_df.loc[mask_shots, 'yCoord'] = \
            [-float(y) for y in game_shots.yCoord]
        # game_shots.netDistance = [-x for x in game_shots.netDistance]
        game_events = events_df.loc[events_df.GameID == game_id]
        mask_events = game_events.index.values
        # mask_events = events_df.loc[events_df.GameID == game_id].index.values
        # event_shots.netDistance = [-x for x in event_shots.netDistance]
        events_df.loc[mask_events, 'xCoord'] = \
            [-float(x) for x in game_events.xCoord]
        events_df.loc[mask_events, 'yCoord'] = \
            [-float(y) for y in game_events.yCoord]
        for j, ind in mask_shots:
            new_x = -float(shots_df.xCoord.iloc[ind])
            new_y = -float(shots_df.yCoord.iloc[ind])
            # this_shot = shots_df.loc[ind]
            shots_df.loc[ind, 'xCoord'] = new_x
            shots_df.loc[ind, 'yCoord'] = new_y
            home_end = shots_df.loc[ind, 'home_end']
            # home_end = this_shot.home_end
            shots_df.loc[ind, 'netDistance'] = calc_coord_diff(new_x, new_y,
                                                               home_end=home_end)
            shots_df.loc[ind, 'netAngle'] = calc_net_angle(new_x, new_y,
                                                           home_end=home_end)
            y_gt_0 = new_y >= 0
            right_shot = shots_df.loc[ind, 'shooterHand'] == 'R'
            shots_df.loc[ind, 'offWingShot'] = home_end ^ (y_gt_0 ^ right_shot)
            # shots_df.loc[ind, 'offWingShot'] = ~this_shot.offWingShot
print(f'It took {timedelta(seconds=(time() - t_start))} to fix the coordinates')

# Load the player data at a dictionary
with open(froot + '/../data/players.csv', 'r') as f:
    dict_reader = csv.DictReader(f)
    player_dict = list(dict_reader)
players = {player_x['PlayerID']: player_x for player_x in player_dict}

# Extract shot events to construct a shot table
shot_types = ['GOAL', 'SHOT', 'MISS', 'BLOCK']

last_event = None
last_game_id = None
last_shot = {}
shot_id = 1
shot_list = []
t_start = time()
print('Starting shot table generation')
for i, event_x in enumerate(all_events):
    # If the event is not a shot, continue
    event_type = event_x['eventTypeId']
    if event_type not in shot_types:
        last_event = event_x.copy()
        continue
    shot_x = event_x.copy()
    shot_x.update(shot_bools.copy())

    # If it is a new game, reset the shot ID
    if last_game_id != event_x['GameID']:
        last_game_id = event_x['GameID']
        shot_id = 1
    shot_x['ShotID'] = shot_id

    # Rename columns and categories
    shot_x['shooterHome'] = shot_x.pop('player1Home')
    if event_type == 'BLOCK':
        shot_x['shooterID'] = int(float(shot_x.pop('player2ID')))
        shot_x['shooterHome'] = not eval(shot_x['shooterHome'])
    else:
        shot_x['shooterID'] = int(float(shot_x.pop('player1ID')))
        shot_x['shooterHome'] = eval(shot_x['shooterHome'])
    shot_x['shotType'] = shot_x.pop('secondaryType')
    shot_x['shotResult'] = shot_x.pop('eventTypeId')

    # Set booleans
    if event_type == 'BLOCK':
        shot_x['blocked'] = True
    elif event_type == 'MISS':
        shot_x['missed'] = True
    elif event_type == 'GOAL':
        shot_x['goal'] = True

    # Add additional information
    period = int(shot_x['period'])
    home_shot = shot_x['shooterHome']
    home_end = bool(home_shot) ^ bool(period % 2 == 1)
    x = float(shot_x['xCoord'])
    y = float(shot_x['yCoord'])
    # shot_x['emptyNet'] = shot_x['player2ID'] is None
    shot_x['netDistance'] = calc_coord_diff(x, y, home_end=home_end)
    shot_x['netAngle'] = calc_net_angle(x, y, home_end=home_end)
    shooter_id = shot_x['shooterID']
    shot_x['shooterHand'] = players[str(shooter_id)]['shootsCatches']
    y_gt_zero = y >= 0
    right_shot = shot_x['shooterHand'] == 'R'
    shot_x['offWingShot'] = home_end ^ (y_gt_zero ^ right_shot)
    # Reference: Off-wing shot (OWS) bool table (home_end = HE, right shot = R)
    # HE  y>=0 R -> OWS
    # _________________
    #  0   0   0  =  0
    #  0   0   1  =  1
    #  0   1   0  =  1
    #  0   1   1  =  0
    #  1   0   0  =  1
    #  1   0   1  =  0
    #  1   1   0  =  0
    #  1   1   1  =  1

    # Calculate values dependent on the previous event
    last_type = last_event['eventTypeId']
    shot_x['lastEventType'] = last_type
    shot_time = game_time_to_sec(event_x['periodTime'])
    last_time = game_time_to_sec(last_event['periodTime'])
    delta_t = shot_time - last_time
    # TODO: Make sure this works for end of period
    #  (it should because period always begins w/ a face-off -> never w/ a shot)
    shot_x['timeSinceLast'] = delta_t
    if last_type in ['SHOT', 'BLOCK'] and delta_t < 3:
        shot_x['reboundShot'] = True
    try:
        last_x, last_y = float(last_event['xCoord']), float(last_event['yCoord'])
    except ValueError:
        print(last_event['GameID'], last_event['EventID'])
    shot_x['lastXCoord'] = last_x
    shot_x['lastYCoord'] = last_y
    shot_x['angleChange'] = calc_angle_diff(last_x, last_y, x, y, home_end)
    shot_x['deltaY'] = calc_coord_diff(last_x, last_x, x, y, home_end,
                                       y_dist=True)

    # Check whether shot leads to a stoppage
    if i < len(all_events) - 1:
        next_type = all_events[i + 1]['eventTypeId']
        shot_x['playEnds'] = next_type == 'STOP'
        reason = all_events[i + 1]['description']
        if next_type == 'STOP' and reason == 'Puck Frozen':
            shot_x['puckFrozen'] = True

    # Pop unwanted columns
    shot_x.pop('EventID', None)
    shot_x.pop('description', None)
    shot_x.pop('periodType', None)
    shot_x.pop('player1ID', None)
    shot_x.pop('player1Type', None)
    shot_x.pop('player2ID', None)
    shot_x.pop('player2Type', None)
    shot_x.pop('assist1ID', None)
    shot_x.pop('assist2ID', None)
    shot_x.pop('PIM', None)

    last_shot = shot_x.copy()
    last_event = event_x.copy()
    shot_list.append(shot_x)
    shot_id += 1

print(f'It took {timedelta(seconds=(time() - t_start))} to create a table '
      f'of {len(shot_list)} shots')


# Save the new shot table and updated event table
new_shot_list = events_df.to_dict('records')
save_nhl_data(froot + '/../data/shots.csv', new_shot_list, overwrite=True)
new_event_list = events_df.to_dict('records')
save_nhl_data(froot + '/../data/game_events.csv', new_event_list, overwrite=True)
