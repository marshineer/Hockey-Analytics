import os
import csv
import pandas as pd
from time import time
from datetime import timedelta, datetime
from nhl_api.ref_common import game_time_to_sec, calc_coord_diff,\
    calc_net_angle, calc_angle_diff
from nhl_api.common import save_nhl_data


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

# Load the player data as a dictionary
with open(froot + '/../data/players.csv', 'r') as f:
    dict_reader = csv.DictReader(f)
    player_list = list(dict_reader)
players = {int(player_x['PlayerID']): player_x for player_x in player_list}

# Load the game data as a dictionary
with open(froot + '/../data/games.csv', 'r') as f:
    dict_reader = csv.DictReader(f)
    game_list = list(dict_reader)
games = {game_x['GameID']: game_x for game_x in game_list}

# Extract shot events to construct a shot table
shot_types = ['GOAL', 'SHOT', 'MISS', 'BLOCK']

last_event = None
last_game_id = None
last_shot = {}
shot_id = 1
home_shot_cnt = 0
away_shot_cnt = 0
shot_list = []
switch_next = False
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
    game_id = event_x['GameID']
    if last_game_id != game_id:
        last_game_id = game_id
        shot_id = 1
        home_shot_cnt = 0
        away_shot_cnt = 0
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
    # home_end = bool(home_shot) ^ bool(period % 2 == 1)
    home_end = home_shot ^ bool(period % 2 == 1)
    shot_x['pulledGoalie'] = (home_shot and event_x['emptyNetHome']) or \
                             (not home_shot and event_x['emptyNetAway'])
    x, y = float(shot_x['xCoord']), float(shot_x['yCoord'])
    shot_x['netDistance'] = calc_coord_diff(x, y, home_end=home_end)
    shot_x['netAngle'] = calc_net_angle(x, y, home_end=home_end)
    shooter_id = shot_x['shooterID']
    shot_x['shooterHand'] = players[shooter_id]['shootsCatches']
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

    # Add shot total and differential prior to current shot
    shot_x['homeShots'] = home_shot_cnt
    shot_x['awayShots'] = away_shot_cnt
    shot_diff = home_shot_cnt - away_shot_cnt
    shot_x['shotLeadPrior'] = shot_diff if home_shot else -shot_diff
    if event_type in ['SHOT', 'GOAL']:
        if home_shot:
            home_shot_cnt += 1
        else:
            away_shot_cnt += 1

    # Add goal differential prior to current shot
    home_score = int(shot_x['homeScore'])
    away_score = int(shot_x['awayScore'])
    # TODO: remove goal update once included in api_common.append_event()
    if event_type == 'GOAL':
        if home_shot:
            home_score -= 1
            shot_x['homeScore'] = home_score
            event_x['homeScore'] = home_score
            # shot_x['goalLeadPrior'] = home_score - away_score
        else:
            away_score -= 1
            shot_x['awayScore'] = away_score
            event_x['awayScore'] = away_score
            # shot_x['goalLeadPrior'] = away_score - home_score
    goal_diff = home_score - away_score
    shot_x['goalLeadPrior'] = goal_diff if home_shot else -goal_diff
    # if event_type == 'GOAL':
    #     shot_x['goalLeadPrior'] -= 1

    # Calculate values dependent on the previous event
    last_type = last_event['eventTypeId']
    shot_x['lastEventType'] = last_type
    shot_x['lastEventPlayer'] = last_event.get('Player1ID', None)
    last_period = int(last_event['period'])
    last_period_time = last_event['periodTime']
    last_time = (last_period - 1) * 20 * 60 + game_time_to_sec(last_period_time)
    # last_time = game_time_to_sec(last_event['periodTime'])
    period = int(event_x['period'])
    period_time = event_x['periodTime']
    shot_time = (period - 1) * 20 * 60 + game_time_to_sec(period_time)
    # shot_time = game_time_to_sec(event_x['periodTime'])
    shot_x['shotTime'] = shot_time
    delta_t = shot_time - last_time
    # if delta_t < 0:
    #     delta_t = shot_time
    shot_x['timeSinceLast'] = delta_t
    if last_type in ['SHOT', 'BLOCK'] and delta_t <= 3:
        shot_x['reboundShot'] = True
    last_x, last_y = float(last_event['xCoord']), float(last_event['yCoord'])
    shot_x['lastXCoord'] = last_x
    shot_x['lastYCoord'] = last_y
    shot_x['angleChange'] = calc_angle_diff(last_x, last_y, x, y, home_end)
    shot_x['distChange'] = calc_coord_diff(last_x, last_y, x, y, home_end)
    shot_x['deltaY'] = calc_coord_diff(last_x, last_y, x, y, home_end,
                                       y_dist=True)
    shot_x['deltaX'] = calc_coord_diff(last_x, last_y, x, y, home_end,
                                       x_dist=True)
    shot_x['lastSameEnd'] = (last_x < -25 and x < -25) or \
                            (last_x > 25 and x > 25)
    # shot_x['crossIcePass'] = shot_x['lastSameEnd'] and shot_x['deltaY'] > 10
    if last_type == 'BLOCK':
        same_team = home_shot ^ eval(last_event['player1Home'])
    else:
        try:
            same_team = not (home_shot ^ eval(last_event['player1Home']))
        except NameError:  # Occurs if last event's player1Home is 'nan'
            same_team = True
    # # Check for nan
    # # https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-values
    # elif float(last_event['player1Home']) == float(last_event['player1Home']):
    #     same_team = not (home_shot ^ eval(last_event['player1Home']))
    # else:
    #     same_team = None
    shot_x['lastTeamSame'] = same_team
    if same_team is None:
        shot_x['lastTurnover'] = False
    else:
        shot_x['lastTurnover'] = (last_type == 'TAKEAWAY' and same_team) or \
                                 (last_type == 'GIVEAWAY' and not same_team)
    game_date = str(games[game_id]['datetime'])
    birthdate = players[shooter_id]['birthDate']
    d1 = datetime.strptime(birthdate, '%Y-%m-%d')
    d2 = datetime.strptime(game_date, '%Y-%m-%dT%H:%M:%SZ')
    shot_x['shooterAge'] = (d2 - d1).total_seconds() / (60 * 60 * 24)
    rookie_season = players[shooter_id].get('rookieSeason')
    if shot_x.get('rookieSeason') is not None:
        shot_x['shooterSeasons'] = int(game_date[:4]) - rookie_season
    else:
        shot_x['shooterSeasons'] = 0
    shot_x['shooterPosition'] = players[shooter_id]['position2']

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
    # shot_x.pop('assist1ID', None)
    # shot_x.pop('assist2ID', None)
    shot_x.pop('PIM', None)

    last_shot = shot_x.copy()
    last_event = event_x.copy()
    shot_list.append(shot_x)
    shot_id += 1

print(f'It took {timedelta(seconds=(time() - t_start))} to create a table '
      f'of {len(shot_list)} shots')

# Save the new shot table and updated event table
t_start = time()
save_nhl_data(froot + '/../data/shots.csv', shot_list, overwrite=True)
save_nhl_data(froot + '/../data/game_events.csv', all_events, overwrite=True)
print(f'It took {timedelta(seconds=(time() - t_start))} to save the files')
