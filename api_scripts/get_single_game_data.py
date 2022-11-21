from time import time
import datetime
from nhl_api.api_parsers import get_game_data
from nhl_api.api_common import get_games_in_season, save_nhl_data


# Specify the game
year = 2019
season = 'R'  # 'R' = regular or 'P' = playoff
game = 1  # for playoff game -> int XYZ: X = round, Y = matchup, Z = game
game_str = str(game).zfill(4)
season_id = '02' if season == 'R' else '03'
game_id = f'{year}{season_id}{game_str}'

game_data = get_game_data([game_id])
t_start = time()
game_list = game_data[0]
shift_list = game_data[1]
event_list = game_data[2]
team_boxscores = game_data[3]
skater_boxscores = game_data[4]
goalie_boxscores = game_data[5]
print(f'Finished the {year}{season} season')
print(f'It took {datetime.timedelta(seconds=(time() - t_start))} to '
      f'scrape game {game_id}')

# f_names1 = ['games', 'shifts', 'game_events', 'team_boxscores',
#             'skater_boxscores', 'goalie_boxscores']
# dict_lists = [game_list, shift_list, event_list, team_boxscores,
#               skater_boxscores, goalie_boxscores]
# for j, dict_list in enumerate(dict_lists):
#     fpath = f'data/individual_games/{f_names1[j]}.csv'
#     save_nhl_data(fpath, dict_list, game_data=True)
