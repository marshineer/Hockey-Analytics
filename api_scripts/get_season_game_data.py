# !/home/marshineer/anaconda3/envs/hockey python3

from time import time, sleep
from datetime import timedelta
import os
from nhl_api.api_parsers import get_game_data
from nhl_api.api_common import get_games_in_season, check_dict_exist,\
    save_nhl_data


# Define the first and last years of the query
first_year = 2010
last_year = 2021
season_types = [['R'], ['P']]
season_name = ['regular', 'playoff']

# Load existing records of players, teams and coaches already exists
# Initialize if files do not exist
all_coaches = check_dict_exist('../data/coaches.csv', 'fullName')
all_teams = check_dict_exist('../data/teams.csv', 'TeamID')
all_players = check_dict_exist('../data/players.csv', 'PlayerID')

# Pull the data for all seasons in range
froot = str(os.path.dirname(__file__))
for year in range(first_year, last_year + 1):
    for i, season_type in enumerate(season_types):
        t_start = time()
        print(f'\nStarting the {year} {season_name[i]} season')
        game_ids = get_games_in_season(year, season_type)
        game_data = get_game_data(game_ids, coaches=all_coaches, teams=all_teams,
                                  players=all_players)
        game_list = game_data[0]
        shift_list = game_data[1]
        event_list = game_data[2]
        team_boxscores = game_data[3]
        skater_boxscores = game_data[4]
        goalie_boxscores = game_data[5]
        print(f'Finished the {year} {season_name[i]} season')
        print(f'It took {timedelta(seconds=(time() - t_start))} to '
              f'scrape {len(game_list)} games')

        f_names1 = ['games', 'shifts', 'game_events', 'team_boxscores',
                    'skater_boxscores', 'goalie_boxscores']
        dict_lists = [game_list, shift_list, event_list, team_boxscores,
                      skater_boxscores, goalie_boxscores]
        for j, dict_list in enumerate(dict_lists):
            fpath = froot + f'/../data/{f_names1[j]}.csv'
            save_nhl_data(fpath, dict_list)

        # Save the coaches, teams and players
        all_coaches_list = [coach for coach in all_coaches.values()]
        all_players_list = [player for player in all_players.values()]
        all_teams_list = [team for team in all_teams.values()]

        f_names2 = ['coaches', 'players', 'teams']
        dict_lists = [all_coaches_list, all_players_list, all_teams_list]
        for j, dict_list in enumerate(dict_lists):
            fpath = froot + f'/../data/{f_names2[j]}.csv'
            save_nhl_data(fpath, dict_list, False)

        # Delay 2 minutes to avoid getting banned by the NHL.com API
        sleep(120)
