import numpy as np
import pandas as pd


# Modelling TODOs
# TODO: Should the rink be divided into zones, or left as a continuous variable?

def calc_travel_dist(city1, city2):
    """ Calculates the distance (km) between two cities.

    The city locations are given in latitude and longitude. This function is
    used to approximate the travel distances between cities for NHL teams. The
    city center is used as the location, rather than the precise location of
    the airport or arena, as I doubt the additional precision will significantly
    affect the model outcome.

    The function uses the Haversine formula
    http://www.movable-type.co.uk/scripts/latlong.html

    Parameters
        city1: tuple = (latitude, longitude) coordinates of the first city
        city2: tuple = (latitude, longitude) coordinates of the second city

    Returns
        dist: int = distance between the two cities (km)
    """

    # Convert the coordinates to radians
    phi1_rad = np.radians(city1[0])
    lambda1_rad = np.radians(city1[1])
    phi2_rad = np.radians(city2[0])
    lambda2_rad = np.radians(city2[1])

    # Calculate the distance using the Haversine formula
    R = 6371  # Earth's radius (km)
    phi_diff = phi2_rad - phi1_rad
    lambda_diff = lambda2_rad - lambda1_rad
    a1 = np.sin(phi_diff / 2)**2
    a2 = np.cos(phi1_rad) * np.cos(phi2_rad) * np.sin(lambda_diff / 2)**2
    c = np.arctan2(np.sqrt(a1 + a2), np.sqrt(1 - (a1 + a2)))
    dist = R * c

    return dist


def binary_encoder(data, col_ls):
    """ Encodes numericized categorical data as a set of binary columns.

    Categorical data needs to be encoded such that it is usable by ML algorithms.
    However, large numbers of categories in one-hot encoding leads to large
    feature dimensions. Therefore, this function uses binary encoding to reduce
    the space required to encode these features. Binary encoding avoids the issue
    of collisions, which occurs with other methods such as hash encoding.

    The length of the binary encoding required is calculated based on the number
    of unique categories for each column. The encoded columns are concatenated
    at the beginning of the data frame.

    This function requires that the categorical data has already been converted
    to be represented by a range of ints. This will be updated in later versions
    of the function.

    Paramters
        data: DataFrame = all data
        col_ls: list = list of categorical column names to be encoded

    Returns
        enc_data: DataFrame = categorical data encoded columns
        n_enc_col: int = total number of encoded columns in output
    """

    n_enc_col = 0
    for i, col in enumerate(col_ls):
        # Find the number of unique categories
        n_cats = len(data[col].unique())

        # Determine the length of the binary encoding
        n_dig = len(format(n_cats - 1, 'b'))
        n_enc_col += n_dig

        # Calculate the encoding for each category
        bin_enc = np.zeros((len(data), n_dig), dtype=int)
        for n in range(n_cats):
            inds = data.loc[data[col] == n].index
            bin_val = [int(x) for x in format(n, f'0{n_dig}b')]
            bin_enc[inds, :] = bin_val

        # Assign new column names
        col_names = []
        for d in range(n_dig):
            col_names.append(f'{col}{d}')

        # Concatenate the encoded columns
        if i == 0:
            enc_data = pd.DataFrame(bin_enc, columns=col_names)
        else:
            enc_data = pd.concat([pd.DataFrame(bin_enc, columns=col_names),
                                  enc_data], axis=1)

    return enc_data, n_enc_col


def sort_game_states(shots, players, return_index=False):
    """ Sorts the events into player strength game state categories.

    For each event, the number of players on each team is checked, and the game
    state determined (5v5, 5v4, 4v5, etc...). Power plays and penalty kills are
    considered separate game states, defined from the perspective of the player
    associated with the event. The events are then sorted. The following game
    state categories exist: Even_5v5, Even_4v4, Even_3v3, PP_5v4, PP_5v3, PK_Xv5.
    Other game states (empty net, 4v3, 3v4) are ignored since they rarely occur.

    Parameters
        shots: list = all shot data
        players: dict = all player data, keyed by player ID
        return_index: bool = return the full dataframes or just the indices

    Returns
        even_5v5:list = even strength shots at 5-on-5
        even_4v4:list = even strength shots at 4-on-4
        even_3v3:list = even strength shots at 3-on-3
        pp_5v4:list = power play shots at 5-on-4
        pp_5v3:list = power play shots at 5-on-3
        pk_Xv5:list = all penalty kill shots
        other:list = all other shots
    """

    even_5v5 = []
    even_4v4 = []
    even_3v3 = []
    pp_5v4 = []
    pp_5v3 = []
    pk_Xv5 = []
    other = []
    for i, shot in enumerate(shots):
        home_players = eval(shot['players_home'])
        n_home = len(home_players)
        away_players = eval(shot['players_away'])
        n_away = len(away_players)
        home_shot = shot['shooter_home']

        # Check for empty net condition
        home_en = True
        away_en = True
        for player in home_players:
            if players[player]['position'] == 'G':
                home_en = False
        for player in away_players:
            if players[player]['position'] == 'G':
                away_en = False
        if home_en or away_en:
            other.append(shot)
            continue

        # Check for other game states
        if n_home == 6 and n_away == 6:
            even_5v5.append(shot)
        elif n_home == 5 and n_away == 5:
            even_4v4.append(shot)
        elif n_home == 4 and n_away == 4:
            even_3v3.append(shot)
        elif home_shot:
            if n_home == 6 and n_away == 5:
                pp_5v4.append(shot)
            elif n_home == 6 and n_away == 4:
                pp_5v3.append(shot)
            elif n_home < 6 and n_away == 6:
                pk_Xv5.append(shot)
            else:
                other.append(shot)
        elif not home_shot:
            if n_away == 6 and n_home == 5:
                pp_5v4.append(shot)
            elif n_away == 6 and n_home == 4:
                pp_5v3.append(shot)
            elif n_away < 6 and n_home == 6:
                pk_Xv5.append(shot)
            else:
                other.append(shot)
        else:
            other.append(shot)

    # Store or reset the indices
    game_states = [even_5v5, even_4v4, even_3v3, pp_5v4, pp_5v3, pk_Xv5, other]
    game_state_lbls = ['5v5', '4v4', '3v3', '5v4', '5v3', 'Xv5', 'other']
    if return_index:
        output = []
        for state in game_states:
            output.append(state.index)
        return output, game_state_lbls
    else:
        return game_states, game_state_lbls
