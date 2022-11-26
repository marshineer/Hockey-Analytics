import requests
from time import sleep
from nhl_api.common import get_venue_coords, convert_height_to_cm
from nhl_api.api_lookups import event_cols, skater_stat_cols, goalie_stat_cols,\
    shot_types, penalty_names


def request_json(url, game_id=None, player_id=None, n_attempt=5):
    """ Returns a JSON of requested information.

    Tries to make a request to the url specified. Also checks for several
    exceptions. If the request is for a game or player, the ID should be
    provided for debugging purposes.

    Note: exceptions return True if the program should re-attempt the request
        and returns False if the program should not continue.

    Parameters
        url: str = the API url to be requested
        game_id: int = the game ID associated with the request
        player_id: int = the player ID associated with the request
        n_attempt: int = number of times to retry a connection/timeout error

    Returns
        response: json = the request response in json format
    """
    # cnt = -1
    while True:
        # cnt += 1
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            # print(f'Tried {cnt} attempts')
        except requests.exceptions.HTTPError as errh:
            print(errh)
            if game_id is not None:
                print(f'Could not find game info for game ID: {game_id}')
            elif player_id is not None:
                print(f'Could not find player info for player ID: {player_id}')
            raise
        except requests.exceptions.ConnectionError as errc:
            print(errc)
            if game_id is not None:
                print(f'Connection issues while pulling game {game_id}')
            elif player_id is not None:
                print(f'Connection issues while pulling player {player_id}')
            if n_attempt > 0:
                n_attempt -= 1
                sleep(1)
                continue
            else:
                raise
        except requests.exceptions.Timeout as errt:
            print(errt)
            if game_id is not None:
                print(f'Timeout while pulling game {game_id}')
            elif player_id is not None:
                print(f'Timeout while pulling player {player_id}')
            if n_attempt > 0:
                n_attempt -= 1
                sleep(1)
                continue
            else:
                raise
        except requests.exceptions.RequestException as err:
            print(err)
            if game_id is not None:
                print(f'Request Exception while pulling game {game_id}')
            elif player_id is not None:
                print(f'Request Exception while pulling player {player_id}')
            raise

        return response.json()


def get_games_in_season(season, game_types=None):
    """ Returns a list of all game IDs in a season.

    Includes preseason, regular season and playoff games, unless
    otherwise specified.

    Parameters
        season: int = season of interest (2010 = 2010-11 season)
        game_types: list = game types to include

    Returns
        game_list: list = all games in the given season
    """

    # Set the start and end dates of the season
    start_date = f'{season}-09-01'
    if season == 2016:
        start_date = '2016-09-30'
    elif season == 2020:
        start_date = '2021-01-01'

    end_date = f'{season + 1}-08-31'
    if season == 2019:
        end_date = '2020-12-31'
    elif season == 2020:
        end_date = '2021-07-31'

    # Set the game types to return
    if game_types is None:
        game_types = ['PR', 'R', 'P']

    # Generate list of games
    # TODO: Delete this line
    # end_date = f'{season}-10-08'
    game_list = get_games_in_range(start_date, end_date, game_types)

    # Remove extra games (round-robin and play-in) during COVID season
    if season == 2019:
        game_list = [game for game in game_list if '20190300' not in str(game)]

    return game_list


def get_games_in_range(start_date, end_date, game_types=None):
    """ Returns a sorted list of all game IDs in a date range.

    Dates must be in the form 'XXXX-XX-XX' = 'year-month-day'. Only returns
    preseason, regular season and playoff type games.

    Parameters
        start_date: str = first date in range
        end_date: str = last date in range
        game_types: list = game types to include

    Returns
        game_list: list = all games in the given range
    """

    # Check that the dates are valid
    # TODO: check that the dates correspond to valid calendar dates (check
    #  number of days per month, leap years, number of digits, etc...).
    #  Probably requires look-up tables.

    # Check that the start date occurs before the end date
    API_URL = f'https://statsapi.web.nhl.com/api/v1/' \
              f'schedule?startDate={start_date}&endDate={end_date}'
    if start_date > end_date:
        print('Invalid start and end dates. '
              'The start date must occur before the end date.')
        return list()
    else:
        games_json = request_json(API_URL)

    # Generate a list of all games in the range
    if game_types is None:
        game_types = ['PR', 'R', 'P']
    game_list = []
    for date in games_json['dates']:
        for game in date['games']:
            if game['gameType'] not in game_types:
                continue
            game_list.append(game['gamePk'])

    game_list.sort()
    return game_list


def get_player_stats(player_id):
    """ Pulls the stats for a particular player.

    These stats may include non-NHL seasons.

    Parameters
        player_id: int = unique player identifier

    Returns
        : json = the player's season by season statistics
    """
    API_URL = f'https://statsapi.web.nhl.com/api/v1/people/{player_id}/' \
              f'stats?stats=yearByYear'
    player_json = request_json(API_URL)
    return player_json


def get_player_info(player_id):
    """ Pulls the biographical info for a particular player.

    Parameters
        player_id: int = unique player identifier

    Returns
        : json = the player's biographical info
    """
    API_URL = f'https://statsapi.web.nhl.com/api/v1/people/{player_id}'
    player_json = request_json(API_URL, player_id=player_id)
    return player_json


def add_team(team_dict, team_key, all_teams):
    """ Adds a new team to the existing dictionary.

    Parameters
        team_dict: dict = basic data for the team
        team_key: str = unique identifier for the team
        all_teams: dict = accumulated data for all teams
    """
    team_x = team_dict['team'].copy()
    team_x.pop('name', None)
    team_x.pop('link', None)
    team_x.pop('triCode', None)
    team_x['TeamID'] = team_x.pop('id')
    all_teams[team_key] = team_x


def update_teams(team_dict, all_teams):
    """ Updates info for newly added teams.

    Parameters
        team_dict: dict = location information for the team
        all_teams: dict = record of metadata for all teams in dataset
    """
    id_key = str(team_dict['id'])
    location = team_dict['venue']['city']
    if 'city' in all_teams[id_key]:
        if all_teams[id_key]['arenaLatitude'] is None:
            venue_lat, venue_lon = get_venue_coords(location)
            all_teams[id_key]['arenaLatitude'] = venue_lon
            all_teams[id_key]['arenaLlongitude'] = venue_lat
        return
    all_teams[id_key]['teamLocation'] = team_dict.get('locationName')
    all_teams[id_key]['teamName'] = team_dict.get('teamName')
    all_teams[id_key]['arenaCity'] = location
    all_teams[id_key]['arenaName'] = team_dict['venue'].get('name')
    venue_lat, venue_lon = get_venue_coords(location)
    all_teams[id_key]['arenaLatitude'] = venue_lon
    all_teams[id_key]['arenaLlongitude'] = venue_lat


def add_player(player_dict, player_key, all_players):
    """ Adds a new player to the existing dictionary.

    Parameters
        player_dict: dict = basic data for the player
        player_id: str = unique player identifier
        all_players: dict = accumulated data for all players
    """
    player_x = player_dict['person'].copy()
    player_x.pop('link', None)
    player_x.pop('rosterStatus', None)
    player_x['PlayerID'] = player_x.pop('id')
    player_x['position'] = player_dict['position'].get('code')
    if player_x['position'] in ['L', 'C', 'R']:
        player_x['position2'] = 'F'
    else:
        player_x['position2'] = player_x['position']
    all_players[player_key] = player_x


def update_players(player_dict, active_players, all_players):
    """ Updates info for newly added players.

    Parameters
        player_dict: dict = physical and biographical data for the player
        all_players: dict = record of metadata for all players in dataset
        active_players: dict = active players for the home and away teams
    """
    player_id = player_dict['id']
    if player_id in active_players:
        id_key = str(player_id)
        if 'birthDate' in all_players[id_key]:
            return
        all_players[id_key]['birthDate'] = player_dict['birthDate']
        if 'nationality' in player_dict:
            all_players[id_key]['nationality'] = player_dict['nationality']
        else:
            all_players[id_key]['nationality'] = player_dict.get('birthCountry')
        if 'height' in player_dict:
            height_cm = convert_height_to_cm(player_dict['height'])
            all_players[id_key]['height_cm'] = height_cm
        else:
            all_players[id_key]['height_cm'] = None
        if 'weight' in player_dict:
            weight_kg = round(player_dict['weight'] / 2.2, 2)
            all_players[id_key]['weight_kg'] = weight_kg
        else:
            all_players[id_key]['weight_kg'] = None


def add_player_season(season, player_id, player_name, player_table, goalie):
    """ Add a single season to a list of player season-by-season stats.

    Parameters
        season: dict = all stats for a single season
        player_id: int = unique player identifier
        player_name: str = full name of player (for table readability)
        player_table: list = accumulated data for all player seasons
        goalie: bool = whether player is a goalie
    """
    if goalie:
        season_x = goalie_stat_cols.copy()
        season_x.update(season['stat'].copy())
        season_x.pop('ties', None)
        season_x.pop('ot', None)
        season_x.pop('savePercentage', None)
        season_x.pop('goalsAgainst', None)
        season_x.pop('goalsAgainstAverage', None)
        season_x.pop('evenStrengthSavePercentage', None)
        season_x.pop('powerPlaySavePercentage', None)
        season_x.pop('shortHandedSavePercentage', None)
    else:
        season_x = skater_stat_cols.copy()
        season_x.update(season['stat'].copy())
        season_x.pop('shotPct', None)
        season_x.pop('shifts', None)
        season_x.pop('timeOnIce', None)
        season_x.pop('penaltyMinutes', None)
    season_x['fullName'] = player_name
    season_x['PlayerID'] = player_id
    season_x['season'] = season['season']
    season_x['TeamID'] = season.get('team', {}).get('id', None)
    season_x['team'] = season['team']['name']
    season_x['league'] = season['league']['name']
    player_table.append(season_x)


def add_coach(coach_dict, coach_ids, team, all_coaches):
    """ Adds a new coach to the existing dictionary.

    Parameters
        coach_dict: dict = basic data for the coach
        coach_ids: dict = coach IDs for the current game
        team: {'home', 'away'} = whether coach is on the home or away team
        all_coaches: dict = accumulated data for all coaches
    """
    coach_x = coach_dict['person'].copy()
    coach_name = coach_x['fullName']
    coach_x.pop('link', None)
    coach_id = len(all_coaches) + 1
    coach_x['CoachID'] = coach_id
    coach_ids[team] = coach_id
    coach_x['code'] = coach_dict['position'].get('code')
    coach_x['position'] = coach_dict['position'].get('type')
    all_coaches[coach_name] = coach_x


def append_team_stats(team_dict, game_id, team, stat_list):
    """ Appends team box score data to a game list

    Parameters
        team_dict: dict = team box score data
        game_id: int = unique game identifier
        team: {'home', 'away'} = whether team is home or away
        stat_list: list = stats for all teams involved in the game
    """
    team_stat_line = team_dict['teamStats']['teamSkaterStats'].copy()
    team_stat_line.pop('powerPlayPercentage', None)
    team_stat_line['GameID'] = game_id
    team_stat_line['TeamID'] = team_dict['team']['id']
    team_stat_line['HomeTeam'] = team == 'home'
    stat_list.append(team_stat_line)


def append_skater_stats(player_stats, game_id, player_id, team, stat_list):
    """ Appends player box score data to a game list

        player_stats: dict = player box score data
        game_id: int = unique game identifier
        player_id: int = unique player identifier
        team: {'home', 'away'} = whether skater is on the home or away team
        stat_list: list = stats for all players involved in the game
    """
    player_stats.pop('faceOffPct', None)
    player_stats['GameID'] = game_id
    player_stats['PlayerID'] = player_id
    player_stats['homeTeam'] = team == 'home'
    stat_list.append(player_stats)


def append_goalie_stats(player_stats, game_id, player_id, team, stat_list):
    """ Appends player box score data to a game list

        player_stats: dict = player box score data
        game_id: int = unique game identifier
        player_id: int = unique player identifier
        team: {'home', 'away'} = whether goalie is on the home or away team
        stat_list: list = stats for all players involved in the game
    """
    player_stats.pop('savePercentage', None)
    player_stats.pop('shortHandedSavePercentage', None)
    player_stats.pop('powerPlaySavePercentage', None)
    player_stats.pop('evenStrengthSavePercentage', None)
    player_stats['GameID'] = game_id
    player_stats['PlayerID'] = player_id
    player_stats['homeTeam'] = team == 'home'
    stat_list.append(player_stats)


def append_event(play, game_id, home_id, event_id, event_list):
    """ Appends play data to a game event list.

    Generates a flat dictionary of data for a single game event (play),
    and appends it to a list of all previous plays in the game.

    Parameters
        play: dict = all data (nested dict) for the game event
        game_id: int = unique game identifier
        home_id: int = unique team identifier of home team
        event_id: int = unique event identifier
        event_list: list = all previous events occurring in game
    #
    # Returns
    #     event_x: dict = extracted data (flat dict) for the game event
    """

    # Extract data common to all events (removing unwanted data)
    event_x = event_cols.copy()
    event_x.update(play['result'].copy())
    event_x.pop('event', None)
    event_x.pop('eventCode', None)
    event_x.pop('penaltySeverity', None)
    event_x.pop('penaltyMinutes', None)
    event_x.pop('strength', None)
    event_x.pop('gameWinningGoal', None)
    event_x['GameID'] = game_id
    event_x['EventID'] = event_id
    event_x['period'] = play['about']['period']
    event_x['periodType'] = play['about']['periodType']
    event_x['periodTime'] = play['about']['periodTime']
    event_x['awayScore'] = play['about']['goals']['away']
    event_x['homeScore'] = play['about']['goals']['home']
    # event_x['awayScore'] = play['about']['goals'].get('away')
    # event_x['homeScore'] = play['about']['goals'].get('home')

    # Skip regular and pre-sesason shootout events
    if str(game_id)[4:6] in ['01', '02'] and event_x['period'] == 5:
        return

    # Extend basic event info
    # event_x.update(event_cols.copy())
    if event_x['eventTypeId'] != 'STOP':
        event_x['xCoord'] = play['coordinates'].get('x')
        event_x['yCoord'] = play['coordinates'].get('y')

    # Add data for particular event types
    event_team_id = play.get('team', {}).get('id', None)
    if 'players' in play:
        player_ids = [player['player']['id'] for player in play['players']]
        event_x['player1ID'] = player_ids[0]
        event_x['player1Type'] = play['players'][0]['playerType']
        event_x['player1Home'] = home_id == event_team_id
    event_type = event_x['eventTypeId']
    if event_type in ['GOAL', 'SHOT']:
        event_x['player2ID'] = player_ids[-1]
        event_x['player2Type'] = play['players'][-1]['playerType']
        if event_type == 'GOAL':
            if len(player_ids) == 1:
                event_x['player2ID'] = None
                event_x['player2Type'] = None
            elif len(player_ids) == 3:
                event_x['assist1ID'] = player_ids[1]
            elif len(player_ids) == 4:
                event_x['assist1ID'] = player_ids[1]
                event_x['assist2ID'] = player_ids[2]
    elif event_type in ['FACEOFF', 'HIT', 'BLOCKED_SHOT', 'PENALTY']:
        if len(player_ids) == 2:
            event_x['player2ID'] = player_ids[1]
            event_x['player2Type'] = play['players'][1]['playerType']
        if event_type == 'PENALTY':
            event_x['PIM'] = play['result']['penaltyMinutes']
    elif event_type == 'GIVEAWAY':
        event_x['player1Type'] = 'Giver'
    elif event_type == 'TAKEAWAY':
        event_x['player1Type'] = 'Taker'

    # Map types to shorter names
    type2 = play['result'].get('secondaryType')
    if event_type in ['GOAL', 'SHOT', 'MISS']:
        event_x['secondaryType'] = shot_types.get(type2, 'OTHER')
        # event_x['secondaryType'] = shot_types[type2]
    elif event_type == 'PENALTY':
        event_x['secondaryType'] = penalty_names.get(type2, 'OTHER')
        # event_x['secondaryType'] = penalty_names[type2]
    else:
        event_x['secondaryType'] = type2
        if event_type == 'BLOCKED_SHOT':
            event_x['eventTypeId'] = 'BLOCK'
        elif event_type == 'MISSED_SHOT':
            event_x['eventTypeId'] = 'MISS'

    event_list.append(event_x)


def extract_game_info(game_data, active_players):
    """ Extracts the basic game data.

    Returns data regarding the game type (regular season, playoff, or preseason),
    location (inc. time zone), home and away teams, and IDs for all participating
    players.

    Parameters
        game_data: dict = json of metadata for a particular game
        active_players: dict = active players for the home and away teams

    Returns
        game_info: dict = reformatted game metadata
    """
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

    return game_info
