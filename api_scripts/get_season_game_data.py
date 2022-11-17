# !/home/marshineer/anaconda3/envs/hockey python3

import csv
from os.path import exists
from time import time, sleep
import datetime
from nhl_api.api_parsers import get_game_data


# Define the first and last years of the query
first_year = 2012
last_year = 2021

# Define the season types
seasons = ['regular', 'playoff']

# Load existing records of players, teams and coaches already exists
# Initialize if files do not exist
try:
    with open(f'../data/coaches.csv', 'r') as f:
        dict_reader = csv.DictReader(f)
        coach_dicts = list(dict_reader)
    all_coaches = {coach_x['fullName']: coach_x for coach_x in coach_dicts}
except FileNotFoundError:
    all_coaches = {}
try:
    with open(f'../data/teams.csv', 'r') as f:
        dict_reader = csv.DictReader(f)
        team_dicts = list(dict_reader)
    all_teams = {team_x['TeamID']: team_x for team_x in team_dicts}
except FileNotFoundError:
    all_teams = {}
try:
    with open(f'../data/players.csv', 'r') as f:
        dict_reader = csv.DictReader(f)
        player_dicts = list(dict_reader)
    all_players = {player_x['PlayerID']: player_x for player_x in player_dicts}
except FileNotFoundError:
    all_players = {}

# Pull the data for all seasons in range
for year in range(first_year, last_year + 1):
    for i, season in enumerate(seasons):
        t_start = time()
        print(f'\nStarting the {year} {season} season')
        game_data = get_game_data(year, season, coaches=all_coaches,
                                  teams=all_teams, players=all_players)
        game_list = game_data[0]
        shift_list = game_data[1]
        event_list = game_data[2]
        team_boxscores = game_data[3]
        skater_boxscores = game_data[4]
        goalie_boxscores = game_data[5]
        print(f'Finished the {year} {season} season')
        print(f'It took {datetime.timedelta(seconds=(time() - t_start))} to '
              f'scrape {len(game_list)} games')

        f_names1 = ['games', 'shifts', 'game_events', 'team_boxscores',
                    'skater_boxscores', 'goalie_boxscores']
        dict_lists = [game_list, shift_list, event_list, team_boxscores,
                      skater_boxscores, goalie_boxscores]
        for j, dict_list in enumerate(dict_lists):
            field_names = dict_list[0].keys()
            fpath = f'data/{f_names1[j]}.csv'
            write_header = not exists(fpath)
            with open(fpath, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                if write_header:
                    writer.writeheader()
                writer.writerows(dict_list)
                csvfile.close()
            write_header = False

        # Save the coaches, teams and players
        all_coaches_list = [coach for coach in all_coaches.values()]
        all_players_list = [player for player in all_players.values()]
        all_teams_list = [team for team in all_teams.values()]

        f_names2 = ['coaches', 'players', 'teams']
        dict_lists = [all_coaches_list, all_players_list, all_teams_list]
        for j, dict_list in enumerate(dict_lists):
            field_names = dict_list[0].keys()
            with open(f'data/{f_names2[j]}.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(dict_list)
                csvfile.close()

        # Delay 2 minutes to avoid getting banned by the NHL.com API
        sleep(120)
