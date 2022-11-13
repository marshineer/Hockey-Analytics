import requests
from nhl_api.parsing_helpers import convert_height_to_cm, get_venue_coords


#  https://stackoverflow.com/questions/6130768/return-a-default-value-if-a-dictionary-key-is-not-available
#  https://realpython.com/python-keyerror/
# TODO: write function for converting shot coordinates to be relative to net locations
#  Should the rink be divided into zones, or left as a continuous variable?
# TODO: figure out how to handle Atlanta (Winnipeg) and Phoenix (Arizona) moving cities
# TODO: add functionality to request games within a date range


event_cols = {'secondaryType': None,
              'player1ID': None,
              'player1Type': None,
              'player2ID': None,
              'player2Type': None,
              'xCoord': None,
              'yCoord': None,
              'assist1ID': None,
              'assist2ID': None,
              'strength': None,
              'emptyNet': None,
              'gameWinningGoal': None,
              'penaltyMinutes': None}


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


def get_game_data(year, season_type, first_game=None, n_games=None,
                  coaches=None, teams=None, players=None):
    """ Requests and formats game data for all games in a given season.

    More description...

    Parameters
        year: int = season during which game occurred
        season_type: {'regular', 'playoffs'} = season type
        first_game: int = first game in the search (min = 1)
        n_games: int = maximum number of games to pull
            regular season: int giving max number of games
            playoffs: int with digits corresponding to max round-matchup-game
        coaches: dict = record of meta-data for all coaches in dataset
        teams: dict = record of meta-data for all teams in dataset
        players: dict = record of meta-data for all players in dataset

    Returns
        game_list: list = meta-data for all games pulled
        shift_info: list = shift data for all players and games
        shot_list: list = shot data for all players and games
        event_list: list = full event data for all games
        team_boxscores: list = home and away box scores for all games
        skater_boxscores: list = home and away skater box scores for all games
        goalie_boxscores: list = home and away goalie box scores for all games
    """

    # Define the NHL.com API url for game data
    API_URL_GAME = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live'
    API_URL_SHIFT = 'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp='\
                    'gameId={}'

    # Define the season type
    seasons_ids = {'regular': '02',
                   'playoffs': '03'}
    season = seasons_ids[season_type]

    # If pulling playoff games, the game ID format is different
    # '0xyz': X = round, Y = matchup, Z = game
    if season_type == 'regular':
        if first_game is None:
            first_game = 1
        if n_games is None:
            n_games = 10000
    elif season_type == 'playoffs':
        if first_game is None:
            first_game = 111
        x_max = 4
        y_max = [8, 4, 2, 1]
        z_max = 7
        if n_games is not None:
            # TODO: handle this better by throwing an error/exception
            if (n_games > 999) | (n_games < 111):
                print("Not a valid 'n_games' value")
                return
            else:
                x_max = min(n_games // 100, x_max)
                y_max = [min((n_games % 100) // 10, ym) for ym in y_max]
                z_max = min(n_games % 10, z_max)

    # Convert the game to a four digit string
    game = str(first_game).zfill(4)

    # Initialize the lists and dictionaries, if necessary
    coaches = {} if coaches is None else coaches
    teams = {} if teams is None else teams
    players = {} if players is None else players
    team_boxscores = []
    skater_boxscores = []
    goalie_boxscores = []
    game_list = []
    event_list = []
    shift_list = []
    shift_dict = {}
    game_dict = {}
    n_games_out = {}

    # Make requests to the NHL.com API for each game of season
    while True:
        # Define the game ID
        # game_id = '{}{}{}'.format(str(year), season, game)
        game_id = f'{str(year)}{season}{game}'
        # print(game_id)

        # Request data for a single game
        game_request_url = API_URL_GAME.format(game_id)
        try:
            game_dict = requests.get(game_request_url, timeout=5)
            game_dict.raise_for_status()
            game_dict = game_dict.json()
        except requests.exceptions.HTTPError as errh:
            print(errh)
            print(f'Could not find game info for game ID: {game_id}')
            return n_games_out
        except requests.exceptions.ConnectionError as errc:
            print(errc)
        except requests.exceptions.Timeout as errt:
            print(errt)
        except requests.exceptions.RequestException as err:
            print(err)

        # Update the game number
        game = str(int(game) + 1).zfill(4)
        if season_type == 'playoffs':
            _, x, y, z = [int(char) for char in game]
            if z > z_max:
                z = 1
                y += 1
                if y > y_max[x - 1]:
                    y = 1
                    x += 1
            game = f'0{x}{y}{z}'

        # If a game record is logged, but the game data does not exist
        if len(game_dict['liveData']['plays']['allPlays']) == 0:
            print(game_id)
            continue

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
        game_list.append(game_output[0])
        event_list += game_output[1]
        team_boxscores += game_output[2]
        skater_boxscores += game_output[3]
        goalie_boxscores += game_output[4]

        # Extract the shift data
        shift_list += parse_shifts(shift_dict)

        # Group outputs in tuple
        n_games_out = (game_list, shift_list, event_list, team_boxscores,
                       skater_boxscores, goalie_boxscores)

        # Check for max number of games
        if season_type == 'regular':
            if (int(game) - int(first_game)) >= n_games:
                return n_games_out
        elif season_type == 'playoffs':
            _, x, y, z = [int(char) for char in game]
            if x > x_max:
                return n_games_out


def game_parser(game_dict, game_id, coaches, teams, players):
    """ Parses the game stats dictionary returned the NHL.com API.

    API endpoint: https://statsapi.web.nhl.com/api/v1/game/gameId/feed/live
    A request to this endpoint returns a dictionary of game stats, including
    the game date and location, teams involved, players involved, game results,
    a list of all plays occurring in the game, and the box score.

    This data is parsed and sorted into data frames to be exported as .csv files.

    Parameters
        game_data: dict = all data pertaining the associated game
        game_id: int = the NHL.com unique game ID
        coaches: dict = record of meta-data for all coaches in dataset
        teams: dict = record of meta-data for all teams in dataset
        players: dict = record of meta-data for all players in dataset

    Returns

    """

    # Separate the types of game data
    boxscore = game_dict['liveData']['boxscore']['teams']
    game_data = game_dict['gameData']
    play_data = game_dict['liveData']['plays']

    # Parse the box score data
    boxscore_output = parse_boxscore(boxscore, game_id, coaches, teams, players)
    team_stats, skater_stats, goalie_stats, active_players = boxscore_output[:-1]
    coach_ids = boxscore_output[-1]

    # Parse the game data and update player info
    game_info = parse_gameData(game_data, teams, players, active_players)
    game_info['shootout'] = game_dict['liveData']['linescore']['hasShootout']
    # game_info['awayCoach'] = boxscore['away']['coaches']['person']['fullName']
    # game_info['homeCoach'] = boxscore['home']['coaches']['person']['fullName']
    game_info['awayCoachID'] = coach_ids['away']
    game_info['homeCoachID'] = coach_ids['home']

    # Parse the game play events
    game_events = parse_liveData(play_data, game_id)

    return game_info, game_events, team_stats, skater_stats, goalie_stats


def parse_boxscore(boxscore, game_id, coaches, teams, players):
    """ Parses the boxscore data from the NHL.com "live" endpoint.

    More description...

    Parameters
        boxscore: dict = box score json for the associated game
        game_id: int = the NHL.com unique game ID
        coaches: dict = record of meta-data for all coaches in dataset
        teams: dict = record of meta-data for all teams in dataset
        players: dict = record of meta-data for all players in dataset

    Returns
        team_stats: list = reformattd team box score stats
        skater_stats: list = reformattd skater box score stats
        goalie_stats: list = reformattd goalie box score stats
        active_players: dict = active players for the home and away teams
    """

    # Initialize the containers
    team_stats = []
    skater_stats = []
    goalie_stats = []
    active_players = {}
    coach_ids = {}

    for i, (key, team_dict) in enumerate(boxscore.items()):
        # Extract team info and box scores
        # Team data
        team_id_str = str(team_dict['team']['id'])
        if team_id_str not in teams:
            team_x = team_dict['team'].copy()
            team_x.pop('link', None)
            team_x.pop('triCode', None)
            team_x['TeamID'] = team_x.pop('id')
            teams[team_id_str] = team_x

        # Team stats
        team_stat_line = team_dict['teamStats']['teamSkaterStats'].copy()
        team_stat_line['GameID'] = game_id
        team_stat_line['TeamID'] = team_dict['team']['id']
        team_stat_line['HomeTeam'] = i
        team_stats.append(team_stat_line)

        # Extract player info and box scores
        skater_stat_list = []
        goalie_stat_list = []
        active_skaters = team_dict['skaters']
        active_goalies = team_dict['goalies']
        for player_dict in team_dict['players'].values():
            player_x = player_dict['person'].copy()
            player_id = player_x['id']

            if player_id in active_skaters:
                # Skater stats (if a players has no stats, they weren't active)
                try:
                    player_stat_line = player_dict['stats']['skaterStats'].copy()
                except KeyError:
                    active_skaters.remove(player_id)
                    continue
                player_stat_line.pop('faceOffPct', None)
                player_stat_line['GameID'] = game_id
                player_stat_line['PlayerID'] = player_id
                player_stat_line['homeTeam'] = i
                skater_stat_list.append(player_stat_line)

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

            elif player_id in active_goalies:
                # Goalie stats (if a players has no stats, they weren't active)
                try:
                    player_stat_line = player_dict['stats']['goalieStats'].copy()
                except KeyError:
                    active_goalies.remove(player_id)
                    continue
                player_stat_line.pop('savePercentage', None)
                player_stat_line.pop('shortHandedSavePercentage', None)
                player_stat_line.pop('powerPlaySavePercentage', None)
                player_stat_line.pop('evenStrengthSavePercentage', None)
                player_stat_line['GameID'] = game_id
                player_stat_line['PlayerID'] = player_id
                player_stat_line['homeTeam'] = i
                goalie_stat_list.append(player_stat_line)

                # Goalie data
                if str(player_id) not in players:
                    player_x.pop('link', None)
                    player_x.pop('rosterStatus', None)
                    player_x['PlayerID'] = player_x.pop('id')
                    player_x['position'] = player_dict['position']['code']
                    players[str(player_id)] = player_x

        # Extract the coach info
        for coach_dict in team_dict['coaches']:
            if coach_dict['position']['code'] != 'HC':
                continue
            coach_x = coach_dict['person'].copy()
            coach_name = coach_x['fullName']
            if coach_name not in coaches:
                coach_x.pop('link', None)
                coach_x['CoachID'] = len(coaches)
                coach_x['code'] = coach_dict['position']['code']
                coach_ids[key] = len(coaches)
                coach_x['position'] = coach_dict['position']['type']
                coaches[coach_name] = coach_x
            else:
                coach_ids[key] = coaches[coach_name]['CoachID']

        # Append the lists for home and away
        active_players[key] = active_skaters + active_goalies
        skater_stats += skater_stat_list
        goalie_stats += goalie_stat_list

    return team_stats, skater_stats, goalie_stats, active_players, coach_ids


def parse_gameData(game_data, teams, players, active_players):
    """ Parses the gameData data from the NHL.com "live" endpoint.

    More description...

    Parameters
        game_data: dict = json of meta-data for a particular game
        teams: dict = record of meta-data for all teams in dataset
        players: dict = record of meta-data for all players in dataset
        active_players: dict = active players for the home and away teams

    Returns
        game_info: dict = reformatted meta-data
    """
    # Extract game data
    game_info = game_data['game'].copy()
    game_info['GameID'] = game_info.pop('pk')
    if game_info['type'] == 'R':
        game_info['type'] = 'REG'
    elif game_info['type'] == 'P':
        game_info['type'] = 'PLA'
    elif game_info['type'] == 'X':
        game_info['type'] = 'PRE'
    else:
        print('Unfamiliar game type')
    game_info['datetime'] = game_data['datetime']['dateTime']
    game_info['awayTeamId'] = game_data['teams']['away']['id']
    game_info['homeTeamId'] = game_data['teams']['home']['id']
    game_info['activeAwayPlayers'] = active_players['away']
    game_info['activeHomePlayers'] = active_players['home']
    game_info['location'] = game_data['teams']['home']['venue']['city']
    game_info['arena'] = game_data['venue']['name']
    game_info['timeZone'] = game_data['teams']['home']['venue']['timeZone']['tz']
    game_info['timeZoneOffset'] = \
        game_data['teams']['home']['venue']['timeZone']['offset']

    # Update team info
    for team_dict in game_data['teams'].values():
        id_key = str(team_dict['id'])
        location = team_dict['venue']['city']
        if 'city' in teams[id_key]:
            continue
        teams[id_key]['teamName'] = team_dict['teamName']
        teams[id_key]['teamLocation'] = team_dict['locationName']
        teams[id_key]['arenaCity'] = location
        teams[id_key]['arenaName'] = team_dict['venue']['name']
        venue_lat, venue_lon = get_venue_coords(location)
        teams[id_key]['arenaLatitude'] = venue_lon
        teams[id_key]['arenaLlongitude'] = venue_lat

    # Update player info
    all_active_players = active_players['away'] + active_players['home']
    for player_dict in game_data['players'].values():
        player_id = player_dict['id']
        if player_id in all_active_players:
            id_key = str(player_id)
            if 'birthDate' in players[id_key]:
                continue
            players[id_key]['birthDate'] = player_dict['birthDate']
            players[id_key]['nationality'] = player_dict['nationality']
            height_cm = convert_height_to_cm(player_dict['height'])
            players[id_key]['height_cm'] = height_cm
            players[id_key]['weight_kg'] = round(player_dict['weight'] / 2.2, 2)
            # TODO: pull rookie seasons from player stats and append to .csv
            #  player file later. this boolean does not account for players who
            #  were rookies prior to the first year pulled
            # if player_dict['rookie']:
            #     players[id_key]['rookieSeason'] = game_info['season']

    return game_info


def parse_liveData(play_data, game_id):
    """ Parses the liveData data from the NHL.com "live" endpoint.

    More description...

    Parameters
        play_data: dict = json of all event data for a particular game
        game_id: int = the NHL.com unique game ID

    Returns
        all_events: list = reformattd game events
        all_shots: list = reformattd record of all shots
    """

    ignore_events = ['GAME_SCHEDULED', 'PERIOD_READY', 'PERIOD_START', 'UNKNOWN',
                     'PERIOD_END', 'PERIOD_OFFICIAL', 'GAME_END', 'FIGHT', 'SUB',
                     'GAME_OFFICIAL', 'SHOOTOUT_COMPLETE', 'OFFICIAL_CHALLENGE',
                     'EARLY_INTERMISSION_START', 'EARLY_INTERMISSION_END',
                     'EMERGENCY_GOALTENDER']

    play_list = play_data['allPlays']
    all_events = []
    # all_shots = []

    event_cnt = 0
    for play in play_list:
        # Skip ignored events
        if play['result']['eventTypeId'] in ignore_events:
            continue

        # Extract data common to all events
        event_x = play['result'].copy()
        event_x.update(event_cols.copy())
        event_x['gameID'] = game_id
        event_x['eventID'] = play['about']['eventIdx']
        event_x['eventID'] = event_cnt
        event_cnt += 1
        event_x.pop('event', None)
        event_x.pop('eventCode', None)
        event_x.pop('penaltySeverity', None)
        event_x['period'] = play['about']['period']
        # TODO: look at what the period and periodType are for OT games
        #  (also in playoffs), do I need periodType?
        # event_x['periodType'] = play['about']['periodType']
        event_x['periodTime'] = play['about']['periodTime']
        event_x['awayScore'] = play['about']['goals']['away']
        event_x['homeScore'] = play['about']['goals']['home']
        if event_x['eventTypeId'] != 'STOP':
            try:
                event_x['xCoord'] = play['coordinates']['x']
                event_x['yCoord'] = play['coordinates']['y']
            except KeyError:
                pass
                # print(event_x['eventTypeId'])
                # print(event_x['description'])
        if 'secondaryType' in play['result']:
            event_x['secondaryType'] = play['result']['secondaryType']

        # Extract data for particular events
        if 'players' in play:
            player_ids = [player['player']['id'] for player in play['players']]
            event_x['player1ID'] = player_ids[0]
            event_x['player1Type'] = play['players'][0]['playerType']
        event_type = event_x['eventTypeId']
        if event_type in ['GOAL', 'SHOT']:
            event_x['player2ID'] = player_ids[-1]
            event_x['player2Type'] = play['players'][-1]['playerType']
            if event_type == 'GOAL':
                event_x['strength'] = play['result']['strength']['code']
                if len(player_ids) == 3:
                    event_x['assist1ID'] = player_ids[1]
                elif len(player_ids) == 4:
                    event_x['assist2ID'] = player_ids[2]
        elif event_type in ['FACEOFF', 'HIT', 'BLOCKED_SHOT', 'PENALTY']:
            if len(player_ids) == 2:
                event_x['player2ID'] = player_ids[1]
                event_x['player2Type'] = play['players'][1]['playerType']

        # # Extract shot data
        # # TODO: create a table of shots with gameID, playerID, shotType(?),
        # #  goal (bool), period, periodTime, xCoord, yCoord, homeScore,
        # #  awayScore, missed (bool), saved (bool), blocked (bool)
        # # TODO: shot and event data have a lot of empty space. is there a better
        # #  way to organize these? Are they going to be tables in the database?
        # if event_type in ['GOAL', 'SHOT', 'MISSED_SHOT', 'BLOCKED_SHOT']:
        # #     shot_x = shot_dict.copy()
        #     shot_x.update(event_x)
        #     if (event_type == 'GOAL') | (event_type == 'SHOT'):
        #         shot_x['shotType'] = event_x['secondaryType']
        #         shot_x['goalieID'] = player_ids[-1]
        #         if event_type == 'GOAL':
        #             shot_x['goal'] = True
        #             shot_x['GWG'] = event_x['gameWinningGoal']
        #             shot_x['EN'] = event_x['emptyNet']
        #             shot_x['strength'] = event_x['strength']['code']
        #             if len(player_ids) == 3:
        #                 shot_x['assist1ID'] = player_ids[1]
        #             elif len(player_ids) == 4:
        #                 shot_x['assist2ID'] = player_ids[2]
        #     elif event_type == 'MISSED_SHOT':
        #         shot_x['miss'] = True
        #     elif event_type == 'BLOCKED_SHOT':
        #         shot_x['block'] = True
        #         shot_x['blockerID'] = player_ids[0]
        #         shot_x['playerID'] = player_ids[-1]
        #     all_shots.append(shot_x)

        # # event_x.update(event_info.copy())
        # # event types [faceoff, giveaway, takeaway, penalty, stoppage, hit, sub?]
        # if event_type in ['GOAL', 'SHOT', 'MISSED_SHOT', 'BLOCKED_SHOT']:
        #     event_x['player1ID'] = player_ids[0]
        #     event_x['player1Type'] = play['players'][0]['PlayerType']
        #     if (event_type == 'GOAL') | (event_type == 'SHOT'):
        #         event_x['secondaryType'] = play['result']['secondaryType']
        #         event_x['goalieID'] = player_ids[-1]
        #         if event_type == 'GOAL':
        #             event_x['goal'] = True
        #             event_x['GWG'] = event_x['gameWinningGoal']
        #             event_x['EN'] = event_x['emptyNet']
        #             event_x['strength'] = event_x['strength']['code']
        #             if len(player_ids) == 3:
        #                 event_x['assist1ID'] = player_ids[1]
        #             elif len(player_ids) == 4:
        #                 event_x['assist2ID'] = player_ids[2]
        #     elif event_type == 'MISSED_SHOT':
        #         event_x['miss'] = True
        #     elif event_type == 'BLOCKED_SHOT':
        #         event_x['block'] = True
        #         event_x['player2ID'] = player_ids[0]
        #         event_x['player2Type'] = play['players'][0]['PlayerType']
        #         event_x['player1ID'] = player_ids[-1]
        #         event_x['player1Type'] = play['players'][-1]['PlayerType']
        # elif event_type in ['FACEOFF', 'HIT', 'PENALTY']:
        #     event_x['player1ID'] = player_ids[0]
        #     event_x['player1Type'] = play['players'][0]['PlayerType']
        #     event_x['player2ID'] = player_ids[1]
        #     event_x['player2Type'] = play['players'][1]['PlayerType']
        #     if event_type == 'PENALTY':
        #         event_x['secondaryType'] = play['result']['secondaryType']
        #         event_x['penaltyMinutes'] = play['result']['penaltyMinutes']
        # # elif event_type == 'HIT':
        # #     pass
        # elif event_type in ['GIVEAWAY', 'TAKEAWAY']:
        #     event_x['player1ID'] = player_ids[0]
        # # elif event_type == 'PENALTY':
        # #     event_x['secondaryType'] = play['result']['secondaryType']
        # #     event_x['penaltyMinutes'] = play['result']['penaltyMinutes']
        # # elif event_type == 'STOP':
        # #     pass

        all_events.append(event_x)

    return all_events


def parse_shifts(shift_data):
    """ Parses the data from the NHL.com shifts endpoint.

    More description...

    Parameters
        shift_data: dict = json of all event data for a particular game

    Returns
        all_shifts: list = reformattd record of all shifts
    """

    shift_list = shift_data['data']
    all_shifts = []
    # Extract data common to all events
    for shift in shift_list:
        if shift['detailCode'] == 0:
            shift_x = {'gameID': shift['gameId'],
                       'playerId': shift['playerId'],
                       'shiftId': shift['shiftNumber'],
                       'period': shift['period'],
                       'startTime': shift['startTime'],
                       'endTime': shift['endTime']}
            all_shifts.append(shift_x)

    return all_shifts
