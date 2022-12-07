import os
import pandas as pd
from time import time
from datetime import timedelta
from nhl_api.common import save_nhl_data


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Initiate start time

# Update teams columns
print('Renaming and ordering teams columns')
t_start = time()
teams_df = pd.read_csv(froot + f'/../data/teams.csv')
team_cols = ['TeamID', 'abbreviation', 'teamLocation', 'teamName', 'arenaCity',
             'arenaName', 'arenaLatitude', 'arenaLlongitude', 'oldTeamID']
teams_df = teams_df[team_cols]
new_team_cols = ['team_id', 'abbreviation', 'team_location', 'team_name',
                 'arena_city', 'arena_name', 'arena_latitude', 'arena_longitude',
                 'old_team_id']
teams_df.columns = new_team_cols
teams_df.sort_values('team_id', inplace=True)
new_teams = teams_df.to_dict('records')
save_nhl_data(froot + f'/../data/teams.csv', new_teams, overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat team columns')

# Update players columns
print('Renaming and ordering players columns')
t_start = time()
players_df = pd.read_csv(froot + f'/../data/players.csv')
player_cols = ['PlayerID', 'fullName', 'birthDate', 'nationality',
               'position', 'position2', 'rookieSeason', 'shootsCatches',
               'height_cm', 'weight_kg']
players_df = players_df[player_cols]
new_player_cols = ['player_id', 'full_name', 'birth_date', 'nationality',
                   'position', 'position2', 'rookie_season', 'shoots_catches',
                   'height_cm', 'weight_kg']
players_df.columns = new_player_cols
# players_df.sort_values('player_id')
players_df.sort_values(['position2', 'full_name'], inplace=True)
new_players = players_df.to_dict('records')
save_nhl_data(froot + f'/../data/players.csv', new_players, overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat player columns')

# Update coaches columns
print('Renaming and ordering coaches columns')
t_start = time()
coaches_df = pd.read_csv(froot + f'/../data/coaches.csv')
coach_cols = ['CoachID', 'fullName', 'code', 'position']
coaches_df = coaches_df[coach_cols]
new_coach_cols = ['coach_id', 'full_name', 'title', 'abbreviation']
coaches_df.columns = new_coach_cols
coaches_df.sort_values('coach_id', inplace=True)
new_coaches = coaches_df.to_dict('records')
save_nhl_data(froot + f'/../data/coaches.csv', new_coaches, overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat coach columns')

# Update games columns
print('Renaming and ordering games columns')
t_start = time()
games_df = pd.read_csv(froot + f'/../data/games.csv')
game_cols = ['GameID', 'season', 'type', 'homeTeamId', 'awayTeamId',
             'homeCoachID', 'awayCoachID', 'homeScore', 'awayScore',
             'numberPeriods', 'overtime', 'shootout', 'location', 'arena',
             'datetime', 'timeZone', 'timeZoneOffset', 'activeHomePlayers',
             'activeAwayPlayers']
games_df = games_df[game_cols]
new_game_cols = ['game_id', 'season', 'type', 'home_team_id', 'away_team_id',
                 'home_coach_id', 'away_coach_id', 'home_score', 'away_score',
                 'number_periods', 'overtime', 'shootout', 'location', 'arena',
                 'datetime', 'time_zone', 'time_zone_offset',
                 'active_home_players', 'active_away_players']
games_df.columns = new_game_cols
games_df.sort_values('game_id', inplace=True)
new_games = games_df.to_dict('records')
save_nhl_data(froot + f'/../data/games.csv', new_games, overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat game columns')

# Update shifts columns
print('Renaming shifts columns')
t_start = time()
shifts_df = pd.read_csv(froot + f'/../data/shifts.csv')
new_shifts_cols = ['game_id', 'player_id', 'shift_id', 'period', 'start_time',
                   'end_time', 'duration']
shifts_df.columns = new_shifts_cols
shifts_df.sort_values(['game_id', 'player_id', 'shift_id'], inplace=True)
new_shifts = shifts_df.to_dict('records')
save_nhl_data(froot + f'/../data/shifts.csv', new_shifts, overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat shift columns')

# Update events columns
print('Renaming and ordering events columns')
t_start = time()
events_df = pd.read_csv(froot + f'/../data/game_events.csv')
event_cols = ['GameID', 'EventID', 'eventTypeId', 'secondaryType', 'description',
              'player1ID', 'player1Type', 'player1Home', 'player2ID',
              'player2Type', 'period', 'periodType', 'periodTime', 'homeScore',
              'awayScore', 'xCoord', 'yCoord', 'assist1ID', 'assist2ID',
              'emptyNet', 'PIM', 'playersHome', 'playersAway']
events_df = events_df[event_cols]
new_event_cols = ['game_id', 'event_id', 'event_type_id', 'secondary_type',
                  'description', 'player1_id', 'player1_type', 'player1_home',
                  'player2_id', 'player2_type', 'period', 'period_type',
                  'period_time', 'home_score', 'away_score', 'x_coord',
                  'y_coord', 'assist1_id', 'assist2_id', 'empty_net', 'pim',
                  'players_home', 'players_away']
events_df.columns = new_event_cols
events_df.sort_values(['game_id', 'event_id'], inplace=True)
new_events = events_df.to_dict('records')
save_nhl_data(froot + f'/../data/game_events.csv', new_events, overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat event columns')

# Update shots columns
print('Renaming and ordering shot columns')
t_start = time()
shots_df = pd.read_csv(froot + f'/../data/shots.csv')
shot_cols = ['GameID', 'ShotID', 'shooterID', 'shotType', 'shotResult', 'period',
             'periodTime', 'homeScore', 'awayScore', 'xCoord', 'yCoord',
             'netDistance', 'netAngle', 'reboundShot', 'lastEventType',
             'timeSinceLast', 'lastXCoord', 'lastYCoord', 'deltaY',
             'angleChange', 'playEnds', 'puckFrozen', 'goal', 'missed',
             'blocked', 'emptyNet', 'shooterHome', 'shooterHand', 'offWingShot',
             'playersHome', 'playersAway']
shots_df = shots_df[shot_cols]
new_shot_cols = ['game_id', 'shot_id', 'shooter_id', 'shot_type', 'shot_result',
                 'period', 'period_time', 'home_score', 'away_score', 'x_coord',
                 'y_coord', 'net_distance', 'net_angle', 'rebound_shot',
                 'last_event_type', 'time_since_last', 'last_x_coord',
                 'last_y_coord', 'delta_y', 'angle_change', 'play_ends',
                 'puck_frozen', 'goal', 'missed', 'blocked', 'empty_net',
                 'shooter_home', 'shooter_hand', 'off_wing_shot',
                 'players_home', 'players_away']
shots_df.columns = new_shot_cols
shots_df.sort_values(['game_id', 'shot_id'], inplace=True)
new_shots = shots_df.to_dict('records')
save_nhl_data(froot + f'/../data/shots.csv', new_shots, overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat shot columns')

# Update team box score columns
print('Renaming and ordering team box score columns')
t_start = time()
team_box_df = pd.read_csv(froot + f'/../data/team_boxscores.csv')
team_box_cols = ['GameID', 'TeamID', 'HomeTeam', 'goals', 'shots', 'hits',
                 'blocked', 'pim', 'powerPlayGoals', 'powerPlayOpportunities',
                 'faceOffWinPercentage', 'takeaways', 'giveaways']
team_box_df = team_box_df[team_box_cols]
new_team_box_cols = ['game_id', 'team_id', 'home_team', 'goals', 'shots', 'hits',
                     'blocks', 'pim', 'ppg', 'power_plays', 'faceoff_pct',
                     'takeaways', 'giveaways']
team_box_df.columns = new_team_box_cols
team_box_df.sort_values(['game_id', 'home_team'], inplace=True)
new_team_box = team_box_df.to_dict('records')
save_nhl_data(froot + f'/../data/team_boxscores.csv', new_team_box,
              overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat team box score '
      f'columns')

# Update skater box score columns
print('Renaming and ordering skater box score columns')
t_start = time()
skater_box_df = pd.read_csv(froot + f'/../data/skater_boxscores.csv')
skater_box_cols = ['GameID', 'PlayerID', 'homeTeam', 'goals', 'assists', 'shots',
                   'hits', 'blocked', 'penaltyMinutes', 'plusMinus',
                   'faceOffWins', 'faceoffTaken', 'takeaways', 'giveaways',
                   'powerPlayGoals', 'powerPlayAssists', 'shortHandedGoals',
                   'shortHandedAssists', 'evenTimeOnIce', 'powerPlayTimeOnIce',
                   'shortHandedTimeOnIce', 'timeOnIce']
skater_box_df = skater_box_df[skater_box_cols]
new_skater_box_cols = ['game_id', 'player_id', 'home_team', 'goals', 'assists',
                       'shots', 'hits', 'blocks', 'pim', 'plus_minus',
                       'faceoff_wins', 'faceoff_taken', 'takeaways', 'giveaways',
                       'ppg', 'ppa', 'shg', 'sha', 'even_toi', 'pp_toi',
                       'sh_toi', 'total_toi']
skater_box_df.columns = new_skater_box_cols
skater_box_df.sort_values(['game_id', 'home_team', 'player_id'], inplace=True)
new_skater_box = skater_box_df.to_dict('records')
save_nhl_data(froot + f'/../data/skater_boxscores.csv', new_skater_box,
              overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat skater box score'
      f' columns')

# # Update skater season columns
# print('Renaming and ordering skater season columns')
# t_start = time()
# skater_season_df = pd.read_csv(froot + f'/../data/skater_seasons.csv')
# skater_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
#                       'season', 'games', 'goals', 'assists', 'points', 'shots',
#                       'hits', 'blocked', 'pim', 'plusMinus', 'faceOffPct',
#                       'powerPlayGoals', 'powerPlayPoints', 'shortHandedGoals',
#                       'shortHandedPoints', 'gameWinningGoals', 'overTimeGoals',
#                       'evenTimeOnIce', 'powerPlayTimeOnIce',
#                       'shortHandedTimeOnIce']
# skater_season_df = skater_season_df[skater_season_cols]
# new_skater_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
#                           'season', 'games', 'goals', 'assists', 'points',
#                           'shots', 'hits', 'blocks', 'PIM', 'plusMinus',
#                           'faceOffPct', 'PPG', 'PPP', 'SHG', 'SHP', 'GWG', 'OTG',
#                           'evenTOI', 'PP_TOI', 'SH_TOI']
# skater_season_df.columns = new_skater_season_cols
# skater_season_df.sort_values(['PlayerID', 'season'], inplace=True)
# new_skater_seasons = skater_season_df.to_dict('records')
# save_nhl_data(froot + f'/../data/skater_seasons.csv', new_skater_seasons,
#               overwrite=True)
# print(f'Took {timedelta(seconds=(time() - t_start))} to reformat skater season '
#       f'columns')

# Update goalie box score columns
print('Renaming and ordering goalie box score columns')
t_start = time()
goalie_box_df = pd.read_csv(froot + f'/../data/goalie_boxscores.csv')
goalie_box_cols = ['GameID', 'PlayerID', 'homeTeam', 'decision', 'saves',
                   'shots', 'evenSaves', 'evenShotsAgainst', 'powerPlaySaves',
                   'powerPlayShotsAgainst', 'shortHandedSaves',
                   'shortHandedShotsAgainst', 'goals', 'assists', 'pim',
                   'timeOnIce']
goalie_box_df = goalie_box_df[goalie_box_cols]
goalie_new_box_cols = ['game_id', 'player_id', 'home_team', 'decision', 'saves',
                       'shots', 'saves_even', 'shots_even', 'saves_pp',
                       'shots_pp', 'saves_sh', 'shots_sh', 'goals', 'assists',
                       'pim', 'total_toi']
goalie_box_df.columns = goalie_new_box_cols
goalie_box_df.sort_values(['game_id', 'home_team', 'total_toi'],
                          ascending=[True, True, False], inplace=True)
new_goalie_box = goalie_box_df.to_dict('records')
save_nhl_data(froot + f'/../data/goalie_boxscores.csv', new_goalie_box,
              overwrite=True)
print(f'Took {timedelta(seconds=(time() - t_start))} to reformat goalie box score'
      f' columns')

# # Update goalie season columns
# print('Renaming and ordering goalie season columns')
# t_start = time()
# goalie_season_df = pd.read_csv(froot + f'/../data/goalie_seasons.csv')
# goalie_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
#                       'season', 'games', 'gamesStarted', 'wins', 'losses',
#                       'saves', 'shotsAgainst', 'shutouts', 'goalAgainstAverage',
#                       'evenSaves', 'evenShots', 'powerPlaySaves',
#                       'powerPlayShots', 'shortHandedSaves', 'shortHandedShots',
#                       'timeOnIce']
# goalie_season_df = goalie_season_df[goalie_season_cols]
# new_goalie_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
#                           'season', 'games', 'starts', 'wins', 'losses', 'saves',
#                           'shots', 'shutouts', 'GAA', 'savesEven', 'shotsEven',
#                           'savesPP', 'shotsPP', 'savesSH', 'shotsSH', 'TOI']
# goalie_season_df.columns = new_goalie_season_cols
# goalie_season_df.sort_values(['PlayerID', 'season'], inplace=True)
# new_goalie_seasons = goalie_season_df.to_dict('records')
# save_nhl_data(froot + f'/../data/goalie_seasons.csv', new_goalie_seasons,
#               overwrite=True)
#
# print(f'\nIt took {timedelta(seconds=(time() - t_start))} to rename all tables')
# print(f'Took {timedelta(seconds=(time() - t_start))} to reformat goalie season '
#       f'columns')
