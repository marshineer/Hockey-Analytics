from nhl_api.api_common import get_games_in_range
from nhl_api.api_parsers import get_game_data


# Specify the date range
start_date = f'2010-09-01'
end_date = '2010-11-01'

game_ids = get_games_in_range(start_date, end_date)
print(game_ids)
game_data = get_game_data(game_ids)
# print(game_data)
