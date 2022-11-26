import numpy as np
from itertools import compress


def game_time_to_sec(time_str):
    """ Converts a string of 'minutes-seconds' to seconds.

    Parameters
        time_str: str = time in 'minutes-seconds'

    Returns
        : int = time in seconds
    """
    min_sec = time_str.split(':')
    return int(min_sec[1]) + int(min_sec[0]) * 60


def create_shift_tables(shift_df, player_list, game_type, period):
    """ Generates tables for which players are on the ice at any given time.

    Two tables are created, one each for the home and away teams. A new table is
    created for each period, as that is the easiest way to filter the shift data.
    The look-up tables are NumPy arrays, whose columns correspond to players that
    are active in the game, with a row for every second of the period. If a
    player is on the ice for a given second, they are represented by a one. All
    other entries are zeros.

    Parameters
        shift_df: DataFrame = all shifts for a particular period of one game
        player_list: list of lists = home and away lists of active players
        game_type: {'PRE', 'REG', 'PLA'} = type of game (determines OT length)
        period: int = period the shifts occur in

    Returns
        shift_tables: list of arrays = on-ice look-up tables for home and away
    """

    # Define the numer of seconds for the period (table length)
    n_sec = 20 * 60
    if period > 3:
        if game_type in ['REG', 'PRE']:
            n_sec = 5 * 60
        elif game_type == 'PLA':
            n_sec = 20 * 60

    # Create shift tables for the home and away players
    shift_tables = []
    for players in player_list:
        shift_table = np.zeros((n_sec, len(players)))
        for i, player_id in enumerate(players):
            player_shifts = shift_df[shift_df.PlayerID == player_id]
            for shift in player_shifts.to_dict('records'):
                start_sec = game_time_to_sec(shift['startTime']) + 1
                length = game_time_to_sec(shift['duration'])
                shift_table[start_sec:start_sec + length, i] = 1
        shift_tables.append(shift_table)

    return shift_tables


def add_players_to_events(event_list, shift_df, games_dict):
    """ Gets the player IDs for all players on the ice at a given time.

    The players are identified as either being on the home or away team.

    Parameters
        event_list: list of dicts = event data pulled from the NHL.com API
        shift_df: DataFrame = shift data pulled from the NHL.com API
        games_dict: dict = all games, keyed by their unique game ID
    """

    player_key = 'players{}'
    old_game_id = None
    old_period = None
    for event in event_list:
        # Extract the event time
        new_game_id = int(event['GameID'])
        new_period = int(event['period'])
        event_sec = game_time_to_sec(event['periodTime'])
        if event_sec in [300, 1200]:
            event_sec -= 1
        # Add a second to separate shifts ending at stoppage
        if event['eventTypeId'] == 'FACEOFF' and event_sec not in [299, 1199]:
            event_sec += 1

        if new_game_id != old_game_id or new_period != old_period:
            # Create a shift table for the period
            game_type = games_dict[new_game_id]['type']
            game_shifts = shift_df[shift_df.GameID == new_game_id]
            period_shifts = game_shifts[game_shifts.period == new_period]
            players = [games_dict[new_game_id]['activeHomePlayers'],
                       games_dict[new_game_id]['activeAwayPlayers']]
            shift_tables = create_shift_tables(period_shifts, players, game_type,
                                               new_period)
            old_game_id = new_game_id
            old_period = new_period

        for shift_table, player_list, team in \
                zip(shift_tables, players, ['Home', 'Away']):
            # Get a mask of the players in the ice
            player_mask = (shift_table[event_sec, :] == 1).tolist()
            on_ice_players = list(compress(player_list, player_mask))
            event_key = player_key.format(team)
            event[event_key] = on_ice_players


def calc_coord_diff(x1, y1, x2=None, y2=None, home_end=True, y_dist=False):
    """ Calculates the distance (ft) between two points on the ice.

    The y_dist boolean is used when calculating the cross-ice distance between
    events. It is used as a measure of how far the goalie's focus must travel
    between a previous event and a shot.

    Note: the "home" net (home team's defensive end in the 1st and 3rd periods)
    is located at the coordinates (89, 0). The coordinates of the "away" net are
    (-89, 0).

    Parameters
        x1: int = x-coordinate (ft) of the first point
        y1: int = y-coordinate (ft) of the first point
        x2: int = x-coordinate (ft) of the second point
        y2: int = y-coordinate (ft) of the second point
        home_end: bool = True if shot taken at the "home" net
        y_dist: bool = True if only returning delta y

    Returns
        : float = distance between the two points
    """

    # If the second point is a net
    if x2 is None:
        x2 = 89 if home_end else -89
        y2 = 0

    if y_dist:
        return abs(y2 - y1)
    else:
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calc_net_angle(x1, y1, home_end=True):
    """ Calculates the angle (in degrees) of an event with respect to a net.

    The angle is measured from the centerline of the net, when viewed from the
    goalie's perspective. Events on the goalie's right have negative angles,
    while events on the left have positive angles.

    Note: the "home" net (home team's defensive end in the 1st and 3rd periods)
    is located at the coordinates (89, 0). The coordinates of the "away" net are
    (-89, 0).

    Parameters
        x1: int = x-coordinate (ft) of the shot location
        y1: int = y-coordinate (ft) of the shot location
        home_end: bool = True if angle calculated wrt the "home" net

    Returns
        : float = angle from centerline of net
    """

    # Net position (reference)
    x_net = 89
    y_net = 0

    # Adjust sign of shot coordinates, depending on end
    x1 = -x1 if not home_end else x1
    y1 = -y1 if home_end else y1

    # Relative shot position
    x_event = x_net - x1
    y_event = y_net - y1

    return np.arctan2(y_event, x_event) * 180 / np.pi


def calc_angle_diff(last_x, last_y, new_x, new_y, home_end=True):
    """ Calculate the angle change between events.

    The individual angle of each event is calculated with respect to the same
    net. Therefore, the absolute difference is simply the angle between them.

    Parameters
        last_x: int = x-coordinate (ft) of the last event
        last_y: int = y-coordinate (ft) of the last event
        new_x: int = x-coordinate (ft) of the new event
        new_y: int = y-coordinate (ft) of the new event
        home_end: bool = True if angle calculated wrt the "home" net

    Returns
        : float = angle between the two events
    """
    last_angle = calc_net_angle(last_x, last_y, home_end=home_end)
    new_angle = calc_net_angle(new_x, new_y, home_end=home_end)

    return abs(new_angle - last_angle)
