from time import time
import datetime
from nhl_api.api_parsers import get_game_data


# Specify the game
year = 2010
season = 'R'  # 'R' = regular or 'P' = playoff
game = 4  # for playoff game -> int XYZ: X = round, Y = matchup, Z = game
game_str = str(game).zfill(4)
season_id = '02' if season == 'R' else '03'
game_id = f'{year}{season_id}{game_str}'

# Useful games
# game_id = '2010020001'  # regular season normal
# game_id = '2010020004'  # regular season OT
# game_id = '2010020008'  # regular season shootout
# game_id = '2010030114'  # playoffs 5 OT
# game_id = '2019020114'
# game_id = '2010020317'  # Early intermission start/end bug
# game_id = '2015020425'  # Early intermission start/end bug
game_id = '2010020124'  # Missing shift data

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
