import os
import csv
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
    last_x, last_y = float(last_event['xCoord']), float(last_event['yCoord'])
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

# Save the new shot table
save_nhl_data(froot + '/../data/shots.csv', shot_list, overwrite=True)
