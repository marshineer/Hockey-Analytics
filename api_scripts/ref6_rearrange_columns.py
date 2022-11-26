import os
import pandas as pd
from nhl_api.common import save_nhl_data


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Update teams columns
teams_df = pd.read_csv(froot + f'/../data/teams.csv')
team_cols = ['TeamID', 'abbreviation', 'teamLocation', 'teamName', 'arenaCity',
             'arenaName', 'arenaLatitude', 'arenaLlongitude', 'oldTeamID']
teams_df = teams_df[team_cols]
teams_df.sort_values('TeamID')
new_teams = teams_df.to_dict('records')
save_nhl_data(froot + f'/../data/teams.csv', new_teams, overwrite=True)

# Update coaches columns
coaches_df = pd.read_csv(froot + f'/../data/coaches.csv')
coach_cols = ['CoachID', 'fullName', 'code', 'position']
coaches_df = coaches_df[coach_cols]
coaches_df.sort_values('CoachID')
new_coaches = coaches_df.to_dict('records')
save_nhl_data(froot + f'/../data/coaches.csv', new_coaches, overwrite=True)

# Update players columns
players_df = pd.read_csv(froot + f'/../data/players.csv')
player_cols = ['PlayerID', 'fullName', 'birthDate', 'nationality',
               'position', 'position2', 'rookieSeason', 'shootsCatches',
               'height_cm', 'weight_kg']
players_df = players_df[player_cols]
players_df.sort_values('PlayerID')
players_df.sort_values(['position2', 'fullName'])
new_players = players_df.to_dict('records')
save_nhl_data(froot + f'/../data/players.csv', new_players, overwrite=True)

f_names1 = ['games', 'shifts', 'game_events', 'team_boxscores',
            'skater_boxscores', 'goalie_boxscores']

# Update events columns
events_df = pd.read_csv(froot + f'/../data/game_events.csv')
print(events_df.columns.tolist())
event_cols = ['GameID', 'EventID', 'eventTypeId', 'secondaryType', 'description',
              'player1ID', 'player1Type', 'player1Home', 'player2ID',
              'player2Type', 'period', 'periodType', 'periodTime', 'homeScore',
              'awayScore', 'xCoord', 'yCoord', 'assist1ID', 'assist2ID',
              'emptyNet', 'PIM', 'playersHome', 'playersAway']
events_df = events_df[event_cols]
events_df.sort_values(['GameID', 'EventID'])
new_events = events_df.to_dict('records')
save_nhl_data(froot + f'/../data/game_events.csv', new_events, overwrite=True)

# Update shots columns
shots_df = pd.read_csv(froot + f'/../data/shots.csv')
print(shots_df.columns.tolist())
shot_cols = ['GameID', 'ShotID', 'shooterID', 'shotType', 'shotResult', 'period',
             'periodTime', 'homeScore', 'awayScore', 'xCoord', 'yCoord',
             'netDistance', 'netAngle', 'reboundShot', 'lastEventType',
             'timeSinceLast', 'lastXCoord', 'lastYCoord', 'deltaY',
             'angleChange', 'playEnds', 'puckFrozen', 'goal', 'missed',
             'blocked', 'emptyNet', 'shooterHome', 'shooterHand', 'offWingShot',
             'playersHome', 'playersAway']
shots_df = shots_df[shot_cols]
shots_df.sort_values(['GameID', 'ShotID'])
new_shots = shots_df.to_dict('records')
save_nhl_data(froot + f'/../data/shots.csv', new_shots, overwrite=True)

# Update games columns
games_df = pd.read_csv(froot + f'/../data/games.csv')
print(games_df.columns.tolist())
# game_cols = ['GameID', 'homeTeamId', 'awayTeamId', 'homeCoachID', 'awayCoachID',
#              'homeScore', 'awayScore', 'numberPeriods', 'overtime', 'shootout',
#              'location', 'arena', 'season', 'type', 'datetime', 'timeZone',
#              'timeZoneOffset', 'activeHomePlayers', 'activeAwayPlayers']
game_cols = ['GameID', 'season', 'type', 'homeTeamId', 'awayTeamId',
             'homeCoachID', 'awayCoachID', 'homeScore', 'awayScore',
             'numberPeriods', 'overtime', 'shootout', 'location', 'arena',
             'datetime', 'timeZone', 'timeZoneOffset', 'activeHomePlayers',
             'activeAwayPlayers']
games_df = games_df[game_cols]
games_df.sort_values('GameID')
new_games = games_df.to_dict('records')
save_nhl_data(froot + f'/../data/games.csv', new_games, overwrite=True)

# Update team box score columns
team_box_df = pd.read_csv(froot + f'/../data/team_boxscores.csv')
print(team_box_df.columns.tolist())
team_box_cols = ['GameID', 'TeamID', 'HomeTeam', 'goals', 'shots', 'blocked',
                 'hits', 'pim', 'powerPlayGoals', 'powerPlayOpportunities',
                 'faceOffWinPercentage', 'takeaways', 'giveaways']
team_box_df = team_box_df[team_box_cols]
team_box_df.sort_values(['GameID', 'HomeTeam'])
new_team_box = team_box_df.to_dict('records')
save_nhl_data(froot + f'/../data/team_boxscores.csv', new_team_box,
              overwrite=True)

# Update skater box score columns
skater_box_df = pd.read_csv(froot + f'/../data/skater_boxscores.csv')
print(sorted(skater_box_df.columns))
skater_box_cols = ['GameID', 'PlayerID', 'homeTeam', 'goals', 'assists', 'shots',
                   'hits', 'blocked', 'takeaways', 'giveaways', 'penaltyMinutes',
                   'plusMinus', 'faceOffWins', 'faceoffTaken', 'powerPlayGoals',
                   'powerPlayAssists', 'shortHandedGoals', 'shortHandedAssists',
                   'evenTimeOnIce', 'powerPlayTimeOnIce', 'shortHandedTimeOnIce',
                   'timeOnIce']
skater_box_df = skater_box_df[skater_box_cols]
new_skater_box_cols = ['GameID', 'PlayerID', 'homeTeam', 'goals', 'assists',
                       'shots', 'hits', 'blocks', 'takeaways', 'giveaways',
                       'PIM', 'plusMinus', 'faceOffWins', 'faceoffTaken',
                       'PPG', 'PPA', 'SHG', 'SHA', 'evenTOI', 'PP_TOI',
                       'SH_TOI', 'TOI']
skater_box_df.columns = new_skater_box_cols
skater_box_df.sort_values(['GameID', 'homeTeam'])
new_skater_box = skater_box_df.to_dict('records')
save_nhl_data(froot + f'/../data/skater_boxscores.csv', new_skater_box,
              overwrite=True)

# Update skater season columns
skater_season_df = pd.read_csv(froot + f'/../data/skater_seasons.csv')
print(sorted(skater_season_df.columns))
skater_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
                      'season', 'games', 'goals', 'assists', 'points', 'shots',
                      'hits', 'blocked', 'pim', 'plusMinus', 'faceOffPct',
                      'powerPlayGoals', 'powerPlayPoints', 'shortHandedGoals',
                      'shortHandedPoints', 'gameWinningGoals', 'overTimeGoals',
                      'evenTimeOnIce', 'powerPlayTimeOnIce',
                      'shortHandedTimeOnIce']
skater_season_df = skater_season_df[skater_season_cols]
new_skater_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
                          'season', 'games', 'goals', 'assists', 'points',
                          'shots', 'hits', 'blocks', 'PIM', 'plusMinus',
                          'faceOffPct', 'PPG', 'PPP', 'SHG', 'SHP', 'GWG', 'OTG',
                          'evenTOI', 'PP_TOI', 'SH_TOI']
skater_season_df.columns = new_skater_season_cols
skater_season_df.sort_values(['PlayerID', 'season'])
new_skater_seasons = skater_season_df.to_dict('records')
save_nhl_data(froot + f'/../data/skater_seasons.csv', new_skater_seasons,
              overwrite=True)

# Update goalie box score columns
goalie_box_df = pd.read_csv(froot + f'/../data/goalie_boxscores.csv')
print(sorted(goalie_box_df.columns))
goalie_box_cols = ['GameID', 'PlayerID', 'homeTeam', 'decision', 'saves',
                   'shots', 'evenSaves', 'evenShotsAgainst', 'powerPlaySaves',
                   'powerPlayShotsAgainst', 'shortHandedSaves',
                   'shortHandedShotsAgainst', 'goals', 'assists', 'pim',
                   'timeOnIce']
goalie_box_df = goalie_box_df[goalie_box_cols]
goalie_new_box_cols = ['GameID', 'PlayerID', 'homeTeam', 'decision', 'saves',
                       'shots', 'savesEven', 'shotsEven', 'savesPP', 'shotsPP',
                       'savesSH', 'shotsSH', 'goals', 'assists', 'PIM', 'TOI']
goalie_box_df.columns = goalie_new_box_cols
goalie_box_df.sort_values(['GameID', 'homeTeam', 'TOI'],
                          ascending=[True, True, False])
new_goalie_box = goalie_box_df.to_dict('records')
save_nhl_data(froot + f'/../data/goalie_boxscores.csv', new_goalie_box,
              overwrite=True)

# Update goalie season columns
goalie_season_df = pd.read_csv(froot + f'/../data/goalie_seasons.csv')
print(sorted(goalie_season_df.columns))
goalie_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
                      'season', 'games', 'gamesStarted', 'wins', 'losses',
                      'saves', 'shotsAgainst', 'shutouts', 'goalAgainstAverage',
                      'evenSaves', 'evenShots', 'powerPlaySaves',
                      'powerPlayShots', 'shortHandedSaves', 'shortHandedShots',
                      'timeOnIce']
goalie_season_df = goalie_season_df[goalie_season_cols]
new_goalie_season_cols = ['PlayerID', 'fullName', 'TeamID', 'team', 'league',
                          'season', 'games', 'starts', 'W', 'L', 'saves',
                          'shots', 'shutouts', 'GAA', 'savesEven', 'shotsEven',
                          'savesPP', 'shotsPP', 'savesSH', 'shotsSH', 'TOI']
goalie_season_df.columns = new_goalie_season_cols
goalie_season_df.sort_values(['PlayerID', 'season'])
new_goalie_seasons = goalie_season_df.to_dict('records')
save_nhl_data(froot + f'/../data/goalie_seasons.csv', new_goalie_seasons,
              overwrite=True)
