import requests
import pytest
import pandas as pd
from nhl_api.api_parsers import parse_boxscore, get_game_data
from nhl_api.api_common import get_games_in_season, get_games_in_range
from nhl_api.ref_common import add_players_to_events, create_shift_tables


# # Game data (11 games)
# start_date = '2010-10-07'
# end_date = '2010-10-08'
# game_ids = get_games_in_range(start_date, end_date)
# game_data = get_game_data(game_ids)
# game_list = game_data[0]
# game_dict = {game_x['GameID']: game_x for game_x in game_list}
# shift_df = pd.DataFrame(game_data[1])
# event_list = game_data[2]
# Single game data
game_id = '2010020008'
game_data = get_game_data([game_id])
game_list = game_data[0]
game_dict = {game_x['GameID']: game_x for game_x in game_list}
shift_df = pd.DataFrame(game_data[1])
event_list = game_data[2]


def get_api_game(game_id, key_list):
    """ Pulls a particular object from the NHL.com API.

    Parameters
        game_id: int = game identifier for NHL.com
        key_list: list = keys required to access object of interest

    Returns
        object of interest
    """

    api_dict = {}
    API_URL_GAME = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live'
    try:
        api_dict = requests.get(API_URL_GAME.format(game_id), timeout=5)
        api_dict.raise_for_status()
        api_dict = api_dict.json()
    except requests.exceptions.HTTPError as errh:
        print(errh)
    except requests.exceptions.ConnectionError as errc:
        print(errc)
    except requests.exceptions.Timeout as errt:
        print(errt)
    except requests.exceptions.RequestException as err:
        print(err)

    for key in key_list:
        api_dict = api_dict[key]

    return api_dict


def test_get_games_in_season():
    seasons = [2010 + i for i in range(12)]
    season_types = ['R', 'P']
    n_games = [(1230, 89), (1230, 86), (720, 86), (1230, 93), (1230, 89),
               (1230, 91), (1230, 87), (1271, 84), (1271, 87), (1082, 86),
               (868, 84), (1312, 89)]
    for i, season in enumerate(seasons):
        for j, season_type in enumerate(season_types):
            n_games_returned = len(get_games_in_season(season, season_type))
            assert n_games_returned == n_games[i][j]


def test_parse_boxscore():
    game_id = 2010021002
    key_list = ['liveData', 'boxscore', 'teams']
    boxscore = get_api_game(game_id, key_list)
    ref_boxscore = boxscore.copy()
    away_skaters = ref_boxscore['away']['skaters']
    home_skaters = ref_boxscore['home']['skaters']
    all_skaters = away_skaters + home_skaters

    boxscore_output = parse_boxscore(boxscore, game_id, {}, {}, {})
    team_stats, skater_stats, goalie_stats, active_players, _ = boxscore_output
    for player in skater_stats:
        assert player['PlayerID'] in all_skaters
        if player['homeTeam'] == 1:
            assert player['PlayerID'] in home_skaters
        else:
            assert player['PlayerID'] in away_skaters


def test_parse_shifts():
    # Test whether the condition 'shift['detailCode'] == 0' catches all relevant
    #  shifts (maybe just randomly sample from all games and see whether
    #  description is always "None")
    pass


def test_create_shift_table():
    game_info = game_list[0]
    game_type = game_info['type']
    f_game_id = game_info['GameID']
    players = [game_info['activeHomePlayers'],
               game_info['activeAwayPlayers']]
    game_shifts = shift_df[shift_df.GameID == f_game_id]
    period_shifts = game_shifts[game_shifts.period == 1]
    shift_tables = create_shift_tables(period_shifts, players, game_type, 1)
    for table in shift_tables:
        for sec in range(table.shape[0]):
            n_players = table[sec, :].sum()
            assert 3 < n_players < 7


def players_on_ice_loops(f_events, f_shifts, f_games):
    """ Gets the player IDs for all players on the ice at a given time.

    The players are identified as either being on the home or away team.

    Parameters
        f_events: list of dicts = event data pulled from the NHL.com API
        f_shifts: DataFrame = shift data pulled from the NHL.com API
        f_games: dict = all games, keyed by their unique game ID
    """

    players_on = {'player1Home': None,
                  'player2Home': None,
                  'player3Home': None,
                  'player4Home': None,
                  'player5Home': None,
                  'player6Home': None,
                  'player1Away': None,
                  'player2Away': None,
                  'player3Away': None,
                  'player4Away': None,
                  'player5Away': None,
                  'player6Away': None}

    for event in f_events:
        # Extract the event time data
        event.update(players_on.copy())
        f_game_id = int(event['GameID'])
        period = int(event['period'])
        event_time = event['periodTime']

        # Filter shift data by game ID and period
        game_info = f_games[f_game_id]
        game_shifts = f_shifts[f_shifts.GameID == f_game_id]
        period_shifts = game_shifts[game_shifts.period == period]

        # Go through the shifts of each individual player
        players = period_shifts.PlayerID.unique().tolist()
        home_cnt = 1
        away_cnt = 1
        player_key = 'player{}{}'
        for player_id in players:
            player_shifts = period_shifts[period_shifts.PlayerID == player_id]
            for shift in player_shifts.to_dict('records'):
                if shift['startTime'] > event_time:
                    continue
                if shift['startTime'] <= event_time < shift['endTime']:
                    if player_id in game_info['activeHomePlayers']:
                        this_player = player_key.format(home_cnt, 'Home')
                        event[this_player] = player_id
                        home_cnt += 1
                        break
                    elif player_id in game_info['activeAwayPlayers']:
                        this_player = player_key.format(away_cnt, 'Away')
                        event[this_player] = player_id
                        away_cnt += 1
                        break
                    else:
                        print('The game and shift data do not match')
            if home_cnt >= 7 and away_cnt >= 7:
                break


def test_players_on_ice():
    table_events = event_list.copy()
    add_players_to_events(table_events, shift_df, game_dict)
    loop_events = event_list.copy()
    players_on_ice_loops(loop_events, shift_df, game_dict)
    player_key_root = 'player{}{}'
    for i, event in enumerate(loop_events):
        for team in ['Home', 'Away']:
            for n in range(1, 7):
                player_key = player_key_root.format(n, team)
                assert event[player_key] == table_events[i][player_key]
