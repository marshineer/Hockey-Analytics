import os
import pandas as pd
from time import time, sleep
from datetime import timedelta
from nhl_api.api_parsers import parse_player_seasons
from nhl_api.api_common import get_player_stats, get_player_info
from nhl_api.common import save_nhl_data


# TODO: Errors in some player's names? -> Not really. Just different spellings.
#  Maybe this causes some issues when combining with other data later?
# 'https://github.com/HarryShomer/Hockey-Scraper/blob/master/hockey_scraper/
#  utils/player_name_fixes.json'

# Load the player list
froot = str(os.path.dirname(__file__))
fpath = froot + f'/../data/players.csv'
players_df = pd.read_csv(fpath)
player_dict = players_df.to_dict('records')
players = {player_x['PlayerID']: player_x for player_x in player_dict}

# Find which players are missing values
missing_vals = {}
missing_ids_set = set()
for key in players_df.columns:
    col_ids = players_df.loc[players_df[key].isna(), 'PlayerID'].tolist()
    missing_vals[key] = col_ids
    missing_ids_set = missing_ids_set.union(set(col_ids))
missing_ids = list(missing_ids_set)

t_start = time()
print('Starting the player updates')
skaters_season_stats = []
goalies_season_stats = []
for player_x in players.values():
    player_id = int(player_x['PlayerID'])
    player_name = player_x['fullName']
    full_player_stats = get_player_stats(player_id)
    all_season_stats = full_player_stats['stats'][0]['splits']
    sleep(1)

    # Create tables of the player's (skater or gaolie) season-by-season stats
    if player_x['position'] == 'G':
        goalies_season_stats += parse_player_seasons(all_season_stats, player_id,
                                                     player_name, True)
    else:
        skaters_season_stats += parse_player_seasons(all_season_stats, player_id,
                                                     player_name)

    # Update missing player details
    if player_id in missing_ids:
        player_info = get_player_info(player_id)
        for key in player_x.keys():
            if player_id in missing_vals[key] and \
                    key in player_info['people'][0].keys():
                player_x[key] = player_info['people'][0][key]

    # Add a rookie season column to the player details
    for season_stats in all_season_stats:
        if season_stats['league']['name'] != 'National Hockey League':
            continue
        elif season_stats['stat']['games'] < 10:
            continue
        else:
            player_x['rookieSeason'] = int(season_stats['season'][:4])

print(f'It took {timedelta(seconds=(time() - t_start))} to '
      f'update {len(players)} players')

all_players_list = [player for player in players.values()]
save_nhl_data(froot + f'/../data/players.csv', all_players_list, overwrite=True)
save_nhl_data(froot + f'/../data/skater_seasons.csv', skaters_season_stats,
              overwrite=True)
save_nhl_data(froot + f'/../data/goalie_seasons.csv', goalies_season_stats,
              overwrite=True)
