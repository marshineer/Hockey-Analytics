import requests
from parsing_helpers import convert_height_to_cm, get_venue_coords


# TODO: create a venue lookup table (dictionary) of all the arena's coordinates
#  (x, y) = (longitude, latitude)
# TODO: create a function to calculate travel distance between arenas
# TODO: handle multiple types of play event dictionaries using one of two methods,
#  try-except or dict.get(key)
#  https://stackoverflow.com/questions/6130768/return-a-default-value-if-a-dictionary-key-is-not-available
#  https://realpython.com/python-keyerror/
# TODO: use switch: ['result']['event'] -> case to handle play parsing
# TODO: write function for converting shot coordinates to be relative to net locations
#  Should the rink be divided into zones, or left as a continuous variable?
# TODO: figure out how to handle Atlanta (Winnipeg) and Phoenix (Arizona) moving cities
# TODO: add functionality to request games within a date range

shot_dict = {'playerID': None,
             'goal': False,
             'shotType': None,
             'assist1ID': None,
             'assist2ID': None,
             'GWG': False,
             'EN': False,
             'goalieID': None,
             'block': False,
             'blockerID': None,
             'strength': None,
             'miss': False}


def request_json(url, **params):
    """ Returns a JSON of requested information.

    Tries to make a request to the url specified. Also checks for several
    exceptions.

    Parameters
        url: str = the API url to be requested
        params: dict = parameters associated with the request

    Returns
        response: json = the request response in json format
    """

    try:
        response = requests.get(url, timeout=5, params=params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print(errh)
    except requests.exceptions.ConnectionError as errc:
        print(errc)
    except requests.exceptions.Timeout as errt:
        print(errt)
    except requests.exceptions.RequestException as err:
        print(err)

    return response.json()


def get_game_data(year, season_type, first_game=1, n_games=None,
                  coaches=None, teams=None, players=None):
    """ Requests and formats game data for all games in a given season.

    More description...

    Parameters
        year: int = season during which game occurred
        season_type: {'regular', 'playoffs'} = season type
        first_game: int = first game in the search (min = 1)
        n_games: int = maximum number of games to pull

    Returns
        game_events: DataFrame? Dictionary? = all recorded game events
        game_meta_data: DataFrame? Dictionary? = data associated with the game
        game_event_types: list = all possible game event types
    """

    # Define the NHL.com API url for game data
    API_URL_GAME = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live'
    API_URL_SHIFT = 'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=' \
                    'gameId={}'

    # Define the season type
    seasons_ids = {'regular': '02',
                   'playoffs': '03'}
    season = seasons_ids[season_type]

    # Convert the game to a four digit string
    game = str(first_game).zfill(4)

    # Initialize the dictionaries, if necessary
    coaches = {} if coaches is None else coaches
    teams = {} if teams is None else teams
    players = {} if players is None else players
    team_boxscores = []
    skater_boxscores = []
    goalie_boxscores = []
    game_list = []
    shot_list = []
    event_list = []
    shift_dict = {}
    game_dict = {}
    n_games_out = {}

    # Make requests to the NHL.com API for each game of season
    while True:
        # Define the game ID
        game_id = '{}{}{}'.format(str(year), season, game)
        print(game_id)

        # Request data for a single game
        game_request_url = API_URL_GAME.format(game_id)
        try:
            game_dict = requests.get(game_request_url, timeout=5)
            game_dict.raise_for_status()
            game_dict = game_dict.json()
        except requests.exceptions.HTTPError as errh:
            print(errh)
            print(f'Could not find game info: Year={year}, Game={game}')
            return n_games_out
        except requests.exceptions.ConnectionError as errc:
            print(errc)
        except requests.exceptions.Timeout as errt:
            print(errt)
        except requests.exceptions.RequestException as err:
            print(err)

        # Request shift data for the same game
        shift_request_url = API_URL_SHIFT.format(game_id)
        try:
            shift_dict = requests.get(shift_request_url, timeout=5)
            shift_dict.raise_for_status()
            shift_dict = shift_dict.json()
        except requests.exceptions.HTTPError as errh:
            print(errh)
        except requests.exceptions.ConnectionError as errc:
            print(errc)
        except requests.exceptions.Timeout as errt:
            print(errt)
        except requests.exceptions.RequestException as err:
            print(err)

        # Extract the single game data
        game_output = game_parser(game_dict, game_id, coaches, teams, players)
        game_list += game_output[0]
        event_list += game_output[1]
        shot_list += game_output[2]
        team_boxscores += game_output[3]
        skater_boxscores += game_output[4]
        goalie_boxscores += game_output[5]

        # Extract the shift data
        shift_info = parse_shifts(shift_dict)

        # Group outputs in tuple
        n_games_out = (game_list, shift_info, shot_list, event_list,
                       team_boxscores, skater_boxscores, goalie_boxscores)

        # Update the game number and check for number of games
        game = str(int(game) + 1).zfill(4)
        if (int(game) - int(first_game)) >= n_games:
            return n_games_out


def game_parser(game_dict, game_id, coaches, teams, players):
    """ Parses the game stats dictionary returned the NHL.com API.

    API endpoint: https://statsapi.web.nhl.com/api/v1/game/gameId/feed/live
    A request to this endpoint returns a dictionary of game stats, including
    the game date and location, teams involved, players involved, game results,
    a list of all plays occurring in the game, and the box score.

    This data is parsed and sorted into data frames to be exported as .csv files.

    Parameters
        game_data: dict = all data pertaining to a particular game
        game_id: int = the NHL.com unique game ID

    Returns

    """

    # Separate the types of game data
    box_score = game_dict['liveData']['boxScore']['teams']
    game_data = game_dict['gameData']
    play_data = game_dict['liveData']['plays']

    # Parse the box score data
    box_score_output = parse_boxScore(box_score, game_id,
                                      coaches, teams, players)
    team_stats, skater_stats, goalie_stats, active_players = box_score_output

    # Parse the game data and update player info
    game_info = parse_gameData(game_data, teams, players, active_players)

    # Parse the game play events
    game_events, shot_events = parse_liveData(play_data, game_id)

    # TODO: get game_events and shot_events in a better format for tables
    return game_info, game_events, shot_events,\
        team_stats, skater_stats, goalie_stats


def parse_boxScore(box_score, game_id, coaches, teams, players):
    # Initialize the containers
    team_stats = []
    skater_stats = []
    goalie_stats = []
    active_players = {}

    for i, (key, team_dict) in enumerate(box_score['teams']):
        # Extract team info and box scores
        # Team data
        team_id_str = str(team_dict['team']['id'])
        if team_id_str not in teams:
            team_x = team_dict['team']
            team_x.pop('link', None)
            team_x.pop('triCode', None)
            team_x['TeamID'] = team_x.pop('id')
            teams[team_id_str] = team_x

        # Team stats
        team_stat_line = team_dict['teamStats']['teamSkaterStats']
        team_stat_line['GameID'] = game_id
        team_stat_line['TeamID'] = team_dict['team']['id']
        team_stat_line['HomeTeam'] = i
        team_stats.append(team_stat_line)

        # Extract player info and box scores
        skater_stat_list = []
        goalie_stat_list = []
        active_skaters = team_dict['skaters']
        active_goalies = team_dict['goalies']
        # active_players[key]['skaters'] = active_skaters
        # active_players[key]['goalies'] = active_goalies
        active_players[key] = active_skaters + active_goalies
        for player_dict in team_dict['players'].value():
            player_x = player_dict['person']
            player_id = player_x['id']

            if player_id in active_skaters:
                # Skater data
                if str(player_id) not in players:
                    player_x.pop('link', None)
                    player_x.pop('rosterStatus', None)
                    player_x['PlayerID'] = player_x.pop('id')
                    player_x['position'] = player_dict['position']['code']
                    if player_x['position'] in ['L', 'C', 'R']:
                        player_x['position2'] = 'F'
                    else:
                        player_x['position2'] = 'D'
                    players[str(player_id)] = player_x

                # Skater stats
                player_stat_line = team_dict['stats']['skaterStats']
                player_stat_line.pop('faceOffPct', None)
                player_stat_line['GameID'] = game_id
                player_stat_line['PlayerID'] = player_id
                player_stat_line['homeTeam'] = i
                skater_stat_list.append(player_stat_line)

            elif player_id in active_goalies:
                # Goalie data
                if str(player_id) not in players:
                    player_x.pop('link', None)
                    player_x.pop('rosterStatus', None)
                    player_x['PlayerID'] = player_x.pop('id')
                    player_x['position'] = player_dict['position']['code']
                    players[str(player_id)] = player_x

                # Goalie stats
                player_stat_line = team_dict['stats']['goalieStats']
                player_stat_line.pop('savePercentage', None)
                player_stat_line.pop('powerPlaySavePercentage', None)
                player_stat_line.pop('evenStrengthSavePercentage', None)
                player_stat_line['GameID'] = game_id
                player_stat_line['PlayerID'] = player_id
                player_stat_line['homeTeam'] = i
                goalie_stat_list.append(player_stat_line)

        # Extract the coach info
        for coach_x in team_dict['coaches']['person']:
            coach_name = coach_x['fullName']
            if coach_name not in coaches:
                coach_x.pop('link', None)
                coach_x['code'] = coach_x['code']
                coach_x['position'] = coach_x['type']
                coaches[coach_name] = coach_x

        # Append the lists for home and away
        skater_stats += skater_stat_list
        goalie_stats += goalie_stat_list

    return team_stats, skater_stats, goalie_stats, active_players


def parse_gameData(game_data, teams, players, active_players):
    # Extract game data
    game_info = game_data['game']
    game_info['GameID'] = game_info.pop('pk')
    if game_info['type'] == 'R':
        game_info['type'] = 'REG'
    elif game_info['type'] == 'P':
        game_info['type'] = 'PLA'
    elif game_info['type'] == 'X':
        game_info['type'] = 'PRE'
    else:
        print('Unfamiliar game type')
    game_info['datetime'] = game_data['datetime']['datetime']
    game_info['awayTeamId'] = game_data['teams']['away']['id']
    game_info['homeTeamId'] = game_data['teams']['home']['id']
    # TODO: should the goalies be separate from the players?
    game_info['activeAwayPlayers'] = active_players['away']
    game_info['activeHomePlayers'] = active_players['home']
    game_info['location'] = game_data['teams']['home']['venue']['city']
    game_info['timeZone'] = game_data['teams']['home']['venue']['tz']
    game_info['timeZoneOffset'] = game_data['teams']['home']['venue']['offset']

    # Update team info
    for team_dict in game_info['teams'].values():
        id_keyy = str(team_dict['id'])
        location = team_dict['venue']['city']
        teams[id_keyy]['city'] = location
        teams[id_keyy]['name'] = team_dict['teamName']
        teams[id_keyy]['arenaName'] = team_dict['venue']['name']
        venue_lat, venue_lon = get_venue_coords(location)
        teams[id_keyy]['arenaCoordinates'] = (venue_lat, venue_lon)

    # Update player info
    all_active_players = active_players['away'] + active_players['home']
    for player_dict in game_info['players'].values():
        player_id = player_dict['id']
        if player_id in all_active_players:
            id_key = str(player_id)
            players[id_key]['birthDate'] = player_dict['birthDate']
            players[id_key]['nationality'] = player_dict['nationality']
            height_cm = convert_height_to_cm(player_dict['height'])
            players[id_key]['height_cm'] = height_cm
            players[id_key]['weight_kg'] = player_dict['weight'] / 2.2
            if player_dict['rookie']:
                players[id_key]['rookieSeason'] = game_info['season']

    return game_info


def parse_liveData(play_data, game_id):
    # ignore_events = ['GAME_SCHEDULED', 'PERIOD_READY', 'PERIOD_START',
    #                  'PERIOD_END', 'PERIOD_OFFICIAL', 'GAME_END', 'UNKNOWN']
    ignore_events = ['GAME_SCHEDULED', 'PERIOD_READY', 'PERIOD_START', 'UNKNOWN',
                     'PERIOD_END', 'PERIOD_OFFICIAL', 'GAME_END', 'FIGHT', 'SUB',
                     'GAME_OFFICIAL', 'SHOOTOUT_COMPLETE', 'OFFICIAL_CHALLENGE',
                     'EARLY_INTERMISSION_START', 'EARLY_INTERMISSION_END',
                     'EMERGENCY_GOALTENDER']
    # goal_inds = play_data['scoringPlays']
    # penalty_inds = play_data['penaltyPlays']
    # period_splits = play_data['playsByPeriod']
    play_list = play_data['plays']
    all_events = []
    all_shots = []

    for play in play_list:
        if play['result']['eventTypeId'] in ignore_events:
            continue
        event_x = play['result']
        event_x['gameID'] = game_id
        event_x['eventID'] = play['about']['eventIdx']
        event_x.pop('event', None)
        event_x.pop('eventCode', None)
        event_x['period'] = play['about']['period']
        # TODO: look at what the period and periodType are for OT games
        #  (also in playoffs), do I need periodType?
        # event_x['periodType'] = play['about']['periodType']
        event_x['periodTime'] = play['about']['periodTime']
        event_x['awayScore'] = play['about']['goals']['away']
        event_x['homeScore'] = play['about']['goals']['home']
        event_x['xCoord'] = play['coordinates']['x']
        event_x['yCoord'] = play['coordinates']['y']
        if 'players' in play:
            player_ids = [player['player']['id'] for player in play['players']]
            event_x['playerID'] = player_ids[0]  # TODO: doesn't work for faceoffs
        # TODO: create a table of shots with gameID, playerID, shotType(?),
        #  goal (bool), period, periodTime, xCoord, yCoord, homeScore,
        #  awayScore, missed (bool), saved (bool), blocked (bool)
        # TODO: shot and event data have a lot of empty space. is there a better
        #  way to organize these? Are they going to be tables in the database?
        event_type = event_x['eventTypeId']
        if event_type in ['GOAL', 'SHOT', 'MISSED_SHOT', 'BLOCKED_SHOT']:
            shot_x = shot_dict.copy()
            shot_x.update(event_x)
            if (event_type == 'GOAL') | (event_type == 'SHOT'):
                shot_x['shotType'] = event_x['secondaryType']
                shot_x['goalieID'] = player_ids[-1]
                if event_type == 'GOAL':
                    shot_x['goal'] = True
                    shot_x['GWG'] = event_x['gameWinningGoal']
                    shot_x['EN'] = event_x['emptyNet']
                    shot_x['strength'] = event_x['strength']['code']
                    if len(player_ids) == 3:
                        shot_x['assist1ID'] = player_ids[1]
                    elif len(player_ids) == 4:
                        shot_x['assist2ID'] = player_ids[2]
            elif event_type == 'MISSED_SHOT':
                shot_x['miss'] = True
            elif event_type == 'BLOCKED_SHOT':
                shot_x['block'] = True
                shot_x['blockerID'] = player_ids[0]
                shot_x['playerID'] = player_ids[-1]
            all_shots.append(shot_x)

        elif event_type == 'FACEOFF':
            # event_x['faceoffWinner'] =
            pass
        elif event_type == 'HIT':
            pass
        elif event_type in ['GIVEAWAY', 'TAKEAWA']:
            pass
        elif event_type == 'PENALTY':
            pass
        elif event_type == 'STOP':
            pass

        all_events.append(event_x)

        # event_types = {
        #     'PERIOD START': 'PSTR',
        #     'FACEOFF': 'FAC',
        #     'BLOCKED SHOT': 'BLOCK',
        #     'GAME END': 'GEND',
        #     'GIVEAWAY': 'GIVE',
        #     'GOAL': 'GOAL',
        #     'HIT': 'HIT',
        #     'MISSED SHOT': 'MISS',
        #     'PERIOD END': 'PEND',
        #     'SHOT': 'SHOT',
        #     'STOPPAGE': 'STOP',
        #     'TAKEAWAY': 'TAKE',
        #     'PENALTY': 'PENL',
        #     'EARLY INT START': 'EISTR',
        #     'EARLY INT END': 'EIEND',
        #     'SHOOTOUT COMPLETE': 'SOC',
        #     'CHALLENGE': 'CHL',
        #     'EMERGENCY GOALTENDER': 'EGPID'
        # }
        # # 'PERIOD READY' & 'PERIOD OFFICIAL'..etc aren't found in html...so get rid of them
        # events_to_ignore = ['PERIOD READY', 'PERIOD OFFICIAL', 'GAME READY', 'GAME OFFICIAL', 'GAME SCHEDULED']

    return all_events, all_shots


def parse_shifts(shift_data):
    """

    :param shift_data:
    :return:
    """
    # TODO: maybe make a row for each player, with a list of start and end time
    #  tuple pairs?
    return shift_data
