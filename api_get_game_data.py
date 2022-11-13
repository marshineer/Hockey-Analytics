# import pandas as pd
import csv
from time import time
import datetime
from nhl_api.api_parsers import get_game_data


# Define the first and last years of the query
first_year = 2010  # First year of recording all shot data
last_year = 2021
year = 2010

# Define the season type
season = 'regular'  # or 'playoffs'

# Define the first game number
first_game = 1
n_games = 2000
# Regular season OT game w/ shootout '2010021002'

game_list = []
shift_list = []
shot_list = []
event_list = []
team_boxscores = []
skater_boxscores = []
goalie_boxscores = []
all_coaches = {}
all_teams = {}
all_players = {}
t_start = time()
game_data = get_game_data(year, season, first_game, n_games, coaches=all_coaches,
                          teams=all_teams, players=all_players)
game_list += game_data[0]
shift_list += game_data[1]
event_list += game_data[2]
team_boxscores += game_data[3]
skater_boxscores += game_data[4]
goalie_boxscores += game_data[5]
# # print(len(all_coaches))
# # print(all_coaches)
# print(len(game_list))
# # print(game_list)
# print(len(event_list))
# # print(event_list)
# print(type(team_boxscores))
# print(len(team_boxscores))
# print(type(skater_boxscores))
# print(len(skater_boxscores))
# print(type(goalie_boxscores))
# print(len(goalie_boxscores))
# TODO: merge dictionaries for each season, or pass dictionaries to be updated
#  passing is probably faster
# https://www.realpythonproject.com/day27-fastest-way-to-combine-dictionaries/
# TODO: add a test file to test api functionality
# TODO: make this just a module that is called in a script, rather than writing
#  mains in the module (keep the project more professional)

all_coaches_list = [coach for coach in all_coaches.values()]
all_players_list = [player for player in all_players.values()]
all_teams_list = [team for team in all_teams.values()]

print(f'It took {datetime.timedelta(seconds=(time() - t_start))} to scrape '
      f'{min(n_games, len(game_list))} games')
f_names = ['coaches', 'players', 'teams', 'games', 'shifts', 'team_boxscores',
           'skater_boxscores', 'goalie_boxscores', 'game_events']
dict_lists = [all_coaches_list, all_players_list, all_teams_list, game_list,
              shift_list, team_boxscores, skater_boxscores, goalie_boxscores,
              event_list]
for i, dict_list in enumerate(dict_lists):
    field_names = dict_list[0].keys()

    with open(f'data/{f_names[i]}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(dict_list)
