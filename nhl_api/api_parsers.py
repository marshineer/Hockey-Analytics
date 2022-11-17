import requests
from time import sleep
from nhl_api.api_common import convert_height_to_cm, get_venue_coords


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
        response = requests.get(url, timeout=30, params=params)
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

    Retrieves data for a number of games in a season (regular or playoffs). The
    number of games can be specified or the full season can be retrieved. Several
    types of game data are pulled, including a game summary, boxscores for teams
    and players, game event data (i.e. shots, goals, penalties, hits, etc...),
    and player shift data. The data for each game is stored in a list, and these
    lists are appended to as more games are retrieved.

    Player, coach and team metadata is also accumulated. However, since this
    information only needs to be recorded once for each entity (coach, player or
    team), this data is imported from existing .csv files and appended to when a
    new entity is found in the game data (i.e. an entry is added only when a new
    player, coach or team is involved in a game).

    Parameters
        year: int = season during which game occurred
        season_type: {'regular', 'playoff'} = season type
        first_game: int = first game in the search (min = 1)
        n_games: int = maximum number of games to pull
            regular season: int giving max number of games
            playoffs: int with digits corresponding to max round-matchup-game
        coaches: dict = record of metadata for all coaches in dataset
        teams: dict = record of metadata for all teams in dataset
        players: dict = record of metadata for all players in dataset

    Returns
        game_list: list = metadata for all games pulled
        shift_info: list = shift data for all players and games
        event_list: list = full event data for all games
        team_boxscores: list = home and away box scores for all games
        skater_boxscores: list = home and away skater box scores for all games
        goalie_boxscores: list = home and away goalie box scores for all games
    """

    # Define the NHL.com API endpoints for game and shift data
    API_URL_GAME = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live'
    API_URL_SHIFT = 'https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp='\
                    'gameId={}'

    # Define the season type
    seasons_ids = {'regular': '02',
                   'playoff': '03'}
    season_id = seasons_ids[season_type]

    # If pulling playoff games, the game ID format is different
    # '0xyz': X = round, Y = matchup, Z = game
    if season_type == 'regular':
        if first_game is None:
            first_game = 1
        if n_games is None:
            n_games = 10000
    elif season_type == 'playoff':
        if first_game is None:
            first_game = 111
        x_max = 4
        y_max = [8, 4, 2, 1]
        z_max = 7
        if n_games is not None:
            if (n_games > 999) | (n_games < 111):
                raise ValueError("Not a valid 'n_games' value")
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
    timeout_err = False
    while True:
        # Sleep to avoid IP getting banned
        if season_type == 'regular':
            if int(game) % 100 == 0:
                sleep(30)
            elif int(game) % 10 == 0:
                sleep(2)

        # Define the game ID
        game_id = f'{str(year)}{season_id}{game}'
        if timeout_err:
            print(f'Parsing game {game_id} again after timeout/lost connection')
            timeout_err = False
        # print(game_id)

        # Request data for a single game
        game_request_url = API_URL_GAME.format(game_id)
        try:
            game_dict = requests.get(game_request_url, timeout=30)
            game_dict.raise_for_status()
            game_dict = game_dict.json()
        except requests.exceptions.HTTPError as errh:
            print(errh)
            print(f'Could not find game info for game ID: {game_id}')
            if season_type == 'regular':
                return n_games_out
            elif season_type == 'playoff':
                _, x, y, z = [int(char) for char in game]
                z = 1
                y += 1
                if y > y_max[x - 1]:
                    y = 1
                    x += 1
                    if x > x_max:
                        return n_games_out
                game = f'0{x}{y}{z}'
                continue
        except requests.exceptions.ConnectionError as errc:
            print(errc)
            print(f'Game {game_id} connection issues while pulling game data')
            timeout_err = True
            continue
        except requests.exceptions.Timeout as errt:
            print(errt)
            print(f'Game {game_id} timed out while pulling game data')
            timeout_err = True
            continue
        except requests.exceptions.RequestException as err:
            print(err)

        # Request shift data for the same game
        shift_request_url = API_URL_SHIFT.format(game_id)
        try:
            shift_dict = requests.get(shift_request_url, timeout=30)
            shift_dict.raise_for_status()
            shift_dict = shift_dict.json()
        except requests.exceptions.HTTPError as errh:
            print(errh)
        except requests.exceptions.ConnectionError as errc:
            print(errc)
            print(f'Game {game_id} connection issues while pulling shift data')
            timeout_err = True
            continue
        except requests.exceptions.Timeout as errt:
            print(errt)
            print(f'Game {game_id} timed out while pulling shift data')
            timeout_err = True
            continue
        except requests.exceptions.RequestException as err:
            print(err)

        # Update the game number
        game = str(int(game) + 1).zfill(4)
        if season_type == 'playoff':
            _, x, y, z = [int(char) for char in game]
            if z > z_max:
                z = 1
                y += 1
                if y > y_max[x - 1]:
                    y = 1
                    x += 1
            game = f'0{x}{y}{z}'

        # If a game record is logged, but the game data does not exist
        game_decision = game_dict['liveData']['decisions']
        if len(game_decision) == 0:
            print(f'No game info for game ID: {game_id}')
            if season_type == 'playoff' and x > x_max:
                return n_games_out
            sleep(1)
            continue

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
        elif season_type == 'playoff':
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
        coaches: dict = record of metadata for all coaches in dataset
        teams: dict = record of metadata for all teams in dataset
        players: dict = record of metadata for all players in dataset

    Returns
        game_info: dict = reformatted game metadata
        game_events: list = reformattd game events
        team_stats: list = boxscore stats for teams
        skater_stats: list = boxscore stats for players
        goalie_stats: list = boxscore stats for goalies
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
    game_info['awayCoachID'] = coach_ids.get('away')
    game_info['homeCoachID'] = coach_ids.get('home')

    # Parse the game play events
    game_events = parse_liveData(play_data, game_id)

    return game_info, game_events, team_stats, skater_stats, goalie_stats


def parse_boxscore(boxscore, game_id, coaches, teams, players):
    """ Parses the boxscore data from the NHL.com "live" endpoint.

    Extracts the team and player boxscore data, summarizing the relevant totals
    for each entity. For example, goals, shots, PIMs, hits, etc... are recorded
    for teams. These values are appended to a list.

    Each active team, player or coach in the game is added to the respective
    metadata dictionary, if it does not already contain an entry for them. This
    is determined by checking the metadata dictiontary for the existence of a
    unique number identifying each team, player and coach.

    Parameters
        boxscore: dict = box score json for the associated game
        game_id: int = the NHL.com unique game ID
        coaches: dict = record of metadata for all coaches in dataset
        teams: dict = record of metadata for all teams in dataset
        players: dict = record of metadata for all players in dataset

    Returns
        team_stats: list = reformattd team box score stats
        skater_stats: list = reformattd skater box score stats
        goalie_stats: list = reformattd goalie box score stats
        active_players: dict = active players for the home and away teams
        coach_ids: dict = coaches for the home and away teams
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
                    player_x['position'] = player_dict['position'].get('code')
                    if player_x['position'] in ['L', 'C', 'R']:
                        player_x['position2'] = 'F'
                    else:
                        player_x['position2'] = player_x['position']
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
                    player_x['position'] = player_dict['position'].get('code')
                    players[str(player_id)] = player_x

        # Extract the coach info
        for coach_dict in team_dict['coaches']:
            if coach_dict['position']['code'] != 'HC':
                continue
            coach_x = coach_dict['person'].copy()
            coach_name = coach_x['fullName']
            if coach_name not in coaches:
                coach_x.pop('link', None)
                coach_id = len(coaches) + 1
                coach_x['CoachID'] = coach_id
                coach_ids[key] = coach_id
                coach_x['code'] = coach_dict['position'].get('code')
                coach_x['position'] = coach_dict['position'].get('type')
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

    Summarizes the game metadata, including features such as the teams invovled,
    the home team, the location of the game venue, etc... Additional information
    is also extracted for teams, players and coaches who are newly added to the
    respective dictionaries. These players are identified using their player IDs,
    provided by the active_players input parameter.

    Parameters
        game_data: dict = json of metadata for a particular game
        teams: dict = record of metadata for all teams in dataset
        players: dict = record of metadata for all players in dataset
        active_players: dict = active players for the home and away teams

    Returns
        game_info: dict = reformatted game metadata
    """
    # Extract game data
    game_info = game_data['game'].copy()
    game_info['GameID'] = game_info.pop('pk')
    if game_info['type'] == 'R':
        game_info['type'] = 'REG'
    elif game_info['type'] == 'P':
        game_info['type'] = 'PLA'
    elif game_info['type'] == 'PR':
        game_info['type'] = 'PRE'
    else:
        print('Unfamiliar game type')
    game_info['datetime'] = game_data['datetime']['dateTime']
    game_info['awayTeamId'] = game_data['teams']['away']['id']
    game_info['homeTeamId'] = game_data['teams']['home']['id']
    game_info['activeAwayPlayers'] = active_players['away']
    game_info['activeHomePlayers'] = active_players['home']
    game_info['location'] = game_data['teams']['home']['venue']['city']
    game_info['arena'] = game_data['venue'].get('name')
    game_info['timeZone'] = \
        game_data['teams']['home']['venue']['timeZone'].get('tz')
    game_info['timeZoneOffset'] = \
        game_data['teams']['home']['venue']['timeZone'].get('offset')

    # Update team info
    for team_dict in game_data['teams'].values():
        id_key = str(team_dict['id'])
        location = team_dict['venue']['city']
        if 'city' in teams[id_key]:
            if teams[id_key]['arenaLatitude'] is None:
                venue_lat, venue_lon = get_venue_coords(location)
                teams[id_key]['arenaLatitude'] = venue_lon
                teams[id_key]['arenaLlongitude'] = venue_lat
            continue
        teams[id_key]['teamName'] = team_dict.get('teamName')
        teams[id_key]['teamLocation'] = team_dict['locationName']
        teams[id_key]['arenaCity'] = location
        teams[id_key]['arenaName'] = team_dict['venue'].get('name')
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
            if 'nationality' in player_dict:
                players[id_key]['nationality'] = player_dict['nationality']
            else:
                players[id_key]['nationality'] = player_dict.get('birthCountry')
            if 'height' in player_dict:
                height_cm = convert_height_to_cm(player_dict['height'])
                players[id_key]['height_cm'] = height_cm
            else:
                players[id_key]['height_cm'] = None
            if 'weight' in player_dict:
                weight_kg = round(player_dict['weight'] / 2.2, 2)
                players[id_key]['weight_kg'] = weight_kg
            else:
                players[id_key]['weight_kg'] = None

    return game_info


def parse_liveData(play_data, game_id):
    """ Parses the liveData data from the NHL.com "live" endpoint.

    Generates a list of game event descriptions. Only game event types that are
    relevant to statistical analysis are included, for example, shots, hits,
    penalties, stoppages, faceoffs, etc... Irrelevant game events are ignored.
    This data includes info which can later be converted into a shot event list.

    Parameters
        play_data: dict = json of all event data for a particular game
        game_id: int = the NHL.com unique game ID

    Returns
        all_events: list = reformattd game events
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
        event_x['GameID'] = game_id
        # event_x['EventID'] = play['about']['eventIdx']
        event_x['EventID'] = event_cnt
        event_cnt += 1
        event_x.pop('event', None)
        event_x.pop('eventCode', None)
        event_x.pop('penaltySeverity', None)
        event_x['period'] = play['about']['period']
        event_x['periodTime'] = play['about']['periodTime']
        event_x['awayScore'] = play['about']['goals'].get('away')
        event_x['homeScore'] = play['about']['goals'].get('home')
        if event_x['eventTypeId'] != 'STOP':
            event_x['xCoord'] = play['coordinates'].get('x')
            event_x['yCoord'] = play['coordinates'].get('y')
        event_x['secondaryType'] = play['result'].get('secondaryType')

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

        all_events.append(event_x)

    return all_events


def parse_shifts(shift_data):
    """ Parses the data from the NHL.com shifts endpoint.

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
            shift_x = {'GameID': shift['gameId'],
                       'PlayerID': shift['playerId'],
                       'ShiftID': shift['shiftNumber'],
                       'period': shift['period'],
                       'startTime': shift['startTime'],
                       'endTime': shift['endTime']}
            all_shifts.append(shift_x)

    return all_shifts
