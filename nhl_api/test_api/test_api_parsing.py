import requests
import pytest
from nhl_api.api_parsers import parse_boxscore


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


def test_parseBoxscore():
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


