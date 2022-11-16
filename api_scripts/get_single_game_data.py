import csv
from os.path import exists
from time import time
import datetime
from nhl_api.api_parsers import get_game_data


# Specify the game
year = 2019
season = 'regular'  # 'regular' or 'playoff'
game = 1086  # for playoff game -> int XYZ: X = round, Y = matchup, Z = game
n_games = 1 if season == 'regular' else 111
season_id = '02' if season == 'regular' else '03'
game_id = f'{year}{season_id}'

game_data = get_game_data(year, season, game, n_games)
t_start = time()
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
    fpath = f'data/individual_games/{f_names1[j]}.csv'
    write_header = not exists(fpath)
    with open(fpath, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        writer.writerows(dict_list)
        csvfile.close()
    write_header = False
