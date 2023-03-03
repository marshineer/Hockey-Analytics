from time import sleep
import nhl_api.api_common as cm


def get_game_data(game_ids, coaches=None, teams=None, players=None):
    """ Requests and formats game data for all games in a given season.

    Retrieves data for a list of game IDs. Several types of game data are pulled,
    including a game summary, box scores for teams and players, game event data
    (i.e. shots, goals, penalties, hits, etc...), and player shift data. The data
    for each game is stored in a list, and these lists are appended to as more
    games are retrieved.

    Player, coach and team metadata is also accumulated. However, since this
    information only needs to be recorded once for each entity (coach, player or
    team), this data is imported from existing .csv files and appended to when a
    new entity is found in the game data (i.e. an entry is added only when a new
    player, coach or team is involved in a game).

    Parameters
        game_ids: int = unique game identifier for all games of interest
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
    n_games_out = {}

    # Make requests to the NHL.com API for each game
    for n, game_id in enumerate(game_ids):
        # Sleep to avoid IP getting banned
        if int(n) % 10 == 0:
            sleep(2)

        # Request data for a single game
        game_request_url = API_URL_GAME.format(game_id)
        game_dict = cm.request_json(game_request_url, game_id=game_id,
                                    n_attempt=10)

        # Request shift data for the same game
        shift_request_url = API_URL_SHIFT.format(game_id)
        shift_dict = cm.request_json(shift_request_url, game_id=game_id,
                                     n_attempt=10)

        # If a game record is logged, but the game data does not exist
        game_decision = game_dict['liveData']['decisions']
        if len(game_decision) == 0:
            print(f'No game info for game ID: {game_id}')
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

    return n_games_out


def game_parser(game_dict, game_id, coaches, teams, players):
    """ Parses the game statistics dictionary returned the NHL.com API.

    API endpoint: https://statsapi.web.nhl.com/api/v1/game/gameId/feed/live
    A request to this endpoint returns a dictionary of game stats, including
    the game date and location, teams involved, players involved, game results,
    a list of all plays occurring in the game, and the box score.

    This data is parsed and sorted into data frames to be exported as .csv files.

    Parameters
        game_data: dict = all data pertaining the associated game
        game_id: int = unique game identifier
        coaches: dict = record of metadata for all coaches in dataset
        teams: dict = record of metadata for all teams in dataset
        players: dict = record of metadata for all players in dataset

    Returns
        game_info: dict = reformatted game metadata
        game_events: list = reformattd game events
        team_stats: list = box score stats for teams
        skater_stats: list = box score stats for players
        goalie_stats: list = box score stats for goalies
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
    end_period = game_dict['liveData']['linescore']['currentPeriod']
    game_info['overtime'] = end_period > 3
    game_info['numberPeriods'] = end_period
    game_info['homeCoachID'] = coach_ids.get('home')
    game_info['awayCoachID'] = coach_ids.get('away')
    home_score = boxscore['home']['teamStats']['teamSkaterStats']['goals']
    game_info['homeScore'] = home_score
    away_score = boxscore['away']['teamStats']['teamSkaterStats']['goals']
    game_info['awayScore'] = away_score
    game_info['shootout'] = game_dict['liveData']['linescore']['hasShootout']
    if game_info['shootout']:
        shootout_info = game_dict['liveData']['linescore']['shootoutInfo']
        game_info['home_win'] = (shootout_info['home']['scores'] >
                                 shootout_info['away']['scores'])
    else:
        game_info['home_win'] = home_score > away_score

    # Parse the game play events
    game_events = parse_liveData(play_data, game_id, game_info['homeTeamId'],
                                 game_info['awayTeamId'])

    return game_info, game_events, team_stats, skater_stats, goalie_stats


def parse_boxscore(boxscore, game_id, coaches, teams, players):
    """ Parses the boxscore data from the NHL.com "live" endpoint.

    Extracts the team and player box score data, summarizing the relevant totals
    for each entity. For example, goals, shots, PIMs, hits, etc... are recorded
    for teams. These values are appended to a list.

    Each active team, player or coach in the game is added to the respective
    metadata dictionary, if it does not already contain an entry for them. This
    is determined by checking the metadata dictiontary for the existence of a
    unique number identifying each team, player and coach.

    Parameters
        boxscore: dict = box score json for the associated game
        game_id: int = unique game identifier
        coaches: dict = record of metadata for all coaches in dataset
        teams: dict = record of metadata for all teams in dataset
        players: dict = record of metadata for all players in dataset

    Returns
        team_stats: list = flattened team box score stats
        skater_stats: list = flattened skater box score stats
        goalie_stats: list = flattened goalie box score stats
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
        # Extract team box score stats and basic info
        cm.append_team_stats(team_dict, game_id, key, team_stats)
        team_id_str = str(team_dict['team']['id'])
        if team_id_str not in teams:
            cm.add_team(team_dict, team_id_str, teams)

        # Extract player box score stats and basic info
        active_skaters = team_dict['skaters']
        active_goalies = team_dict['goalies']
        for player_dict in team_dict['players'].values():
            player_id = player_dict['person']['id']

            if player_id in active_skaters:
                # Skater stats (if a players has no stats, they weren't active)
                try:
                    skater_stat_line = player_dict['stats']['skaterStats'].copy()
                except KeyError:
                    active_skaters.remove(player_id)
                    continue
                cm.append_skater_stats(skater_stat_line, game_id, player_id, key,
                                       skater_stats)

                # Add new skater if they are not already in the dict
                if str(player_id) not in players:
                    cm.add_player(player_dict, str(player_id), players, game_id)

            elif player_id in active_goalies:
                # Goalie stats (if a players has no stats, they weren't active)
                try:
                    goalie_stat_line = player_dict['stats']['goalieStats'].copy()
                except KeyError:
                    active_goalies.remove(player_id)
                    continue
                cm.append_goalie_stats(goalie_stat_line, game_id, player_id, key,
                                       goalie_stats)

                # Add new goalie if they are not already in the dict
                if str(player_id) not in players:
                    cm.add_player(player_dict, str(player_id), players, game_id)

        # Extract the coach info
        for coach_dict in team_dict['coaches']:
            if coach_dict['position']['code'] != 'HC':
                continue
            coach_x = coach_dict['person'].copy()
            coach_name = coach_x['fullName']
            if coach_dict['person']['fullName'] not in coaches:
                cm.add_coach(coach_dict, coach_ids, key, coaches)
            else:
                coach_ids[key] = coaches[coach_name]['CoachID']

        # List of active player IDs for home and away teams
        active_players[key] = active_skaters + active_goalies

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
    game_info = cm.extract_game_info(game_data, active_players)

    # Update team info
    for team_dict in game_data['teams'].values():
        cm.update_teams(team_dict, teams)

    # Update player info
    all_active_players = active_players['away'] + active_players['home']
    for player_dict in game_data['players'].values():
        cm.update_players(player_dict, all_active_players, players)

    return game_info


def parse_liveData(play_data, game_id, home_id, away_id):
    """ Parses the liveData data from the NHL.com "live" endpoint.

    Generates a list of game event descriptions. Only game event types that are
    relevant to statistical analysis are included, for example, shots, hits,
    penalties, stoppages, faceoffs, etc... Irrelevant game events are ignored.
    This data includes info which can later be converted into a shot event list.

    Parameters
        play_data: dict = json of all event data for a particular game
        game_id: int = unique game identifier
        home_id: int = unique team identifier of home team
        away_id: int = unique team identifier of away team

    Returns
        all_events: list = reformattd game events
    """

    ignore_events = ['GAME_SCHEDULED', 'PERIOD_READY', 'PERIOD_START', 'UNKNOWN',
                     'PERIOD_END', 'PERIOD_OFFICIAL', 'GAME_END', 'FIGHT', 'SUB',
                     'GAME_OFFICIAL', 'SHOOTOUT_COMPLETE', 'OFFICIAL_CHALLENGE',
                     'EARLY_INTERMISSION_START', 'EARLY_INTERMISSION_END',
                     'EMERGENCY_GOALTENDER', 'EARLY_INT_START', 'EARLY_INT_END',
                     'CHALLENGE']

    all_events = []
    event_id = 0
    for play in play_data['allPlays']:
        # Skip ignored events
        if play['result']['eventTypeId'] in ignore_events:
            continue

        # Extract event information
        cm.append_event(play, game_id, home_id, away_id, event_id, all_events)
        event_id += 1

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
        # TODO: Make sure this if-statement catches all shifts of interest
        if shift['detailCode'] == 0:
            shift_x = {'GameID': shift['gameId'],
                       'PlayerID': shift['playerId'],
                       'ShiftID': shift['shiftNumber'],
                       'period': shift['period'],
                       'startTime': shift['startTime'],
                       'endTime': shift['endTime'],
                       'duration': shift['duration']}
            all_shifts.append(shift_x)

    return all_shifts


def parse_player_seasons(season_list, player_id, player_name, goalie=False):
    """ Populate a player's season-by-season stats table.

    Returns a list of flat dictionaries representing the player's season stats.
    The stats may include leagues other than the NHL, although these contain
    fewer stat categories.

    Parameters
        season_list: list = all seasons of recorded stats for the player
        player_id: int = unique player identifier
        player_name: str = full name of player (for table readability)
        goalie: bool = whether player is a goalie

    Returns
        player_table: list = reformatted season-by-season player stats
    """

    all_seasons = []
    for season in season_list:
        n_games = season['stat'].get('games')
        if n_games is None or n_games == 0:
            continue
        elif goalie and season['stat'].get('wins') is None:
            continue
        cm.add_player_season(season, player_id, player_name, all_seasons, goalie)

    return all_seasons
