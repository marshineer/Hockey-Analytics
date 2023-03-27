import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold


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
        cats = data[col].unique()
        n_cats = len(cats)

        # Determine the length of the binary encoding
        n_dig = len(format(n_cats - 1, 'b'))
        n_enc_col += n_dig

        # Calculate the encoding for each category
        bin_enc = np.zeros((len(data), n_dig), dtype=int)
        for n, cat in enumerate(cats):
            inds = data.loc[data[col] == cat].index
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


def sort_game_states(shots, return_index=False, return_model_data=True):
    """ Sorts the events into player strength game state categories.

    For each event, the number of players on each team is checked, and the game
    state determined (5v5, 5v4, 4v5, etc...). Power plays and penalty kills are
    considered separate game states, defined from the perspective of the player
    associated with the event. The events are then sorted. The following game
    state-strength categories exist: Even_5v5, Even_4v4, Even_3v3, PP_5v4,
    PP_5v3, PP_4v3, PK_4v5, PK_3v5, PK_3v4, empty_net.

    Game states where the goalie is pulled for the attacking team are not
    considered different than they would be without the goalie pulled. This
    is because the shooting percentage with and without the goalie pulled is
    essentially the same. Instead, the pulled goalie is indicated by a boolean
    feature.

    Parameters
        shots: list = all shot data
        return_index: bool = return complete shot data or just the indices
        return_model_data: bool = only return game states used for xG modelling

    Returns
        game_states: list of lists = shots corresponding to all game states
        game_state_lbls: list = shots taken on empty nets
        game_strength_lbls: dict = all other shots
    """

    # Sort shots into game states and define strength boolean features
    even_shots = []
    pp_shots = []
    pk_shots = []
    en_shots = []
    other_shots = []
    for ind, shot in enumerate(shots):
        home_shot = shot['shooter_home']
        en_home = shot['empty_net_home']
        en_away = shot['empty_net_away']
        n_home, n_away = calc_on_ice_players(shot)
        state, strength = get_game_state(n_home, n_away, en_home, en_away,
                                         home_shot)

        if state == 'empty_net':
            en_shots.append(ind)
        elif state == 'Even':
            if strength == '5v5':
                shot.update({'5v5': 1, '4v4': 0, '3v3': 0})
                shot['stratify'] = '100' + ('1' if shot['goal'] == 1 else '0')
            elif strength == '4v4':
                shot.update({'5v5': 0, '4v4': 1, '3v3': 0})
                shot['stratify'] = '010' + ('1' if shot['goal'] == 1 else '0')
            elif strength == '3v3':
                shot.update({'5v5': 0, '4v4': 0, '3v3': 1})
                shot['stratify'] = '001' + ('1' if shot['goal'] == 1 else '0')
            even_shots.append(ind)
        elif state == 'PP':
            if strength == '5v4':
                shot.update({'5v4': 1, '5v3': 0, '4v3': 0})
                shot['stratify'] = '100' + ('1' if shot['goal'] == 1 else '0')
            elif strength == '5v3':
                shot.update({'5v4': 0, '5v3': 1, '4v3': 0})
                shot['stratify'] = '010' + ('1' if shot['goal'] == 1 else '0')
            elif strength == '4v3':
                shot.update({'5v4': 0, '5v3': 0, '4v3': 1})
                shot['stratify'] = '001' + ('1' if shot['goal'] == 1 else '0')
            pp_shots.append(ind)
        elif state == 'PK':
            if strength == '4v5':
                shot.update({'4v5': 1, '3v5': 0, '3v4': 0})
                shot['stratify'] = '100' + ('1' if shot['goal'] == 1 else '0')
            elif strength == '3v5':
                shot.update({'4v5': 0, '3v5': 1, '3v4': 0})
                shot['stratify'] = '010' + ('1' if shot['goal'] == 1 else '0')
            elif strength == '3v4':
                shot.update({'4v5': 0, '3v5': 0, '3v4': 1})
                shot['stratify'] = '001' + ('1' if shot['goal'] == 1 else '0')
            pk_shots.append(ind)
        else:
            other_shots.append(ind)

    # Return list of shot dictionaries, rather than indices
    if not return_index:
        even_shots = [shots[i] for i in even_shots]
        pp_shots = [shots[i] for i in pp_shots]
        pk_shots = [shots[i] for i in pk_shots]
        en_shots = [shots[i] for i in en_shots]
        other_shots = [shots[i] for i in other_shots]

    # Return all game state shots or only those relevant for modelling
    if return_model_data:
        game_states = [even_shots, pp_shots, pk_shots]
        game_state_lbls = ['Even', 'PP', 'PK']
    else:
        game_states = [even_shots, pp_shots, pk_shots, en_shots, other_shots]
        game_state_lbls = ['Even', 'PP', 'PK', 'EN', 'Other']
    game_strength_lbls = {'Even': ['5v5', '4v4', '3v3'],
                          'PP': ['5v4', '5v3', '4v3'],
                          'PK': ['4v5', '3v5', '3v4']}

    return game_states, game_state_lbls, game_strength_lbls


def calc_on_ice_players(shot):
    """ Calculate the number of on-ice players for each team for a given shot.

    Parameters
        shot: dict = single shot data
        players: dict = all player data, keyed by their player ID

    Returns
        n_home: int = number of players on ice for the home team
        n_away: int = number of players on ice for the away team
        empty_net: bool = True if at least one net is empty
        en_home: bool = True if at the home team's net is empty
        en_away: bool = True if at the away team's net is empty
    """
    home_players = eval(shot['players_home'])
    n_home = len(home_players)
    away_players = eval(shot['players_away'])
    n_away = len(away_players)

    return n_home, n_away


def get_game_state(n_home, n_away, en_home, en_away, home_shot):
    """ Determines the game state given number of players and empty nets.

    Parameters
        n_home: int = number of on-ice players for the home team
        n_away: int = number of on-ice players for the away team
        en_home: bool = whether the home team's goalie is pulled
        en_away: bool = whether the away team's goalie is pulled
        home_shot: bool = whether the shooter is on the home team

    Returns
        state: str = game state for the given configuration
        strength: str = teams strengths for the given state
    """
    # TODO: maybe remove the empty net and pulled goalie states?
    #  (these can be calculated in the higher level function)
    if (home_shot and en_away) or (not home_shot and en_home):
        return 'empty_net', ''
    # elif (home_shot and en_home) or (not home_shot and en_away):
    #     return 'pulled_goalie', ''
    elif n_home == n_away:
        if n_home == 6 and n_away == 6:
            return 'Even', '5v5'
        elif n_home == 5 and n_away == 5:
            return 'Even', '4v4'
        elif n_home == 4 and n_away == 4:
            return 'Even', '3v3'
    elif home_shot:
        if n_home > n_away:
            if n_home == 6 and n_away == 5:
                return 'PP', '5v4'
            elif n_home == 6 and n_away == 4:
                return 'PP', '5v3'
            elif n_home == 5 and n_away == 4:
                return 'PP', '4v3'
        else:
            if n_home == 5 and n_away == 6:
                return 'PK', '4v5'
            elif n_home == 4 and n_away == 6:
                return 'PK', '3v5'
            elif n_home == 4 and n_away == 5:
                return 'PK', '3v4'
    elif not home_shot:
        if n_home < n_away:
            if n_away == 6 and n_home == 5:
                return 'PP', '5v4'
            elif n_away == 6 and n_home == 4:
                return 'PP', '5v3'
            elif n_away == 5 and n_home == 4:
                return 'PP', '4v3'
        else:
            if n_away == 5 and n_home == 6:
                return 'PK', '4v5'
            elif n_away == 4 and n_home == 6:
                return 'PK', '3v5'
            elif n_away == 4 and n_home == 5:
                return 'PK', '3v4'


def split_train_test_sets(data_df, cont_feats, bool_feats, target='goal',
                          stratify_feat=None, shuffle=True, test_frac=0.15,
                          shuffle_seed=66, train_seasons=None, test_seasons=None,
                          return_scaler=False):
    """ Split data with a binary target feature into training and test sets.

    Select the required features, split the data into training and test sets,
    and scale the continuous features to the range [0, 1].

    If the data is shuffled, the training and test sets are stratified.
    Stratification means the samples will be distributed such that the chosen
    stratification feature will be distributed among the training and test sets
    according to the ratio set by the test fraction. If stratified by the target
    class, the frequency of the target class follows this ratio. If a feature is
    specified, the distribution of classes feature can be created that allows
    the stratification to incorporate multiple feature columns.

    Rather than shuffling, a list of seasons may be specified for the training
    and test sets. In this case, the data is unshuffled and returned in the
    order specified by the respective season lists.

    Parameters
        data_df: dataframe = all data required to train and test the model
        cont_feats: list = input columns containing continuous features
        bool_feats: list = input columns containing boolean features
        target: str = name of the target feature
        stratify_feat: str = name of the feature used for stratification
        shuffle: bool = whether to split the data by randomly shuffling it
        test_frac: float = fraction of data used for testing
        shuffle_seed: int = random generator seed used for reproducibility
        train_seasons: list = seasons to use for training
        test_seasons: list = seasons to use for testing
        return_scaler: bool = whether to return the sklearn MinMaxScaler

    Returns
        X_train: array = training data (samples, features)
        X_test: array = test data (samples, features)
        y_train: array = training labels (samples)
        y_test: array = test labels (samples)
        x_scaler: MinMaxScaler = scalter fit using training data
    """

    # Split the data
    input_cols = cont_feats + bool_feats
    if shuffle:
        if stratify_feat is None:
            strat_col = data_df[target]
        else:
            strat_col = data_df[stratify_feat]
        input_df = data_df[input_cols]
        target_df = data_df[target]
        split = train_test_split(input_df, target_df, test_size=test_frac,
                                 random_state=shuffle_seed, stratify=strat_col)
        X_train, X_test, y_train, y_test = split
    else:
        train_df = data_df[data_df.season.isin(train_seasons)][input_cols]
        X_train = train_df[input_cols]
        y_train = train_df[target]
        test_df = data_df[data_df.season.isin(test_seasons)][input_cols]
        X_test = test_df[input_cols]
        y_test = test_df[target]

    # Reset the indices
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Scale the data
    X_train, x_scaler = normalize_continuous(X_train, cont_feats)
    X_test = normalize_continuous(X_test, cont_feats, scaler=x_scaler)[0]

    if return_scaler:
        return X_train, X_test, y_train, y_test, x_scaler
    else:
        return X_train, X_test, y_train, y_test


def normalize_continuous(data_df, continuous_feats, scaler=None):
    """ Scale a selection of continuous features to the range [0, 1]

    Parameters
        data_df: dataframe = all feature data for a training set
        continuous_feats: list = continuous feature columns to scale
        scaler: MinMaxScaler = previously fit scaler

    Returns
        norm_df: dataframe = the original data with select features normalized
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data_df[continuous_feats].values)

    train_norm = scaler.transform(data_df[continuous_feats].values)
    data_df[continuous_feats] = train_norm

    return data_df, scaler


def create_stratify_feat(stratify_cols):
    """ Creates a string identifier used to stratify along multiple columns.

    Stratification allows the distribution of a particular feature (or
    combination of features) to follow a given ratio. It is used when a dataset
    is divided into groups, for example, during cross-validation training or
    when splitting the data into training and test sets. The output of this
    function may be passed to sklearn functions which allow stratification,
    enabling stratification across multiple features. Two examples of functions
    that accept this as an input are StratifiedKFold.split() and
    train_test_split()

    Ref: https://www.geeksforgeeks.org/python-convert-list-characters-string/

    Parameters
        stratify_cols: series = feature combination used for stratification

    Returns
        : str = string identifier used to stratify the data
    """
    col_names = stratify_cols.index.tolist()
    stratify_chars = [str(stratify_cols[col]) for col in col_names]
    stratify_string = ''

    return stratify_string.join(stratify_chars)


def add_xg_features(shots_df):
    """ Add features used to predict expected goals.

    Parameters
        shots_df: dataframe = all shot data

    Returns
        shots_df: dataframe = updated shot data
    """
    forward_mask = shots_df.shooter_position == 'F'
    shots_df['forward_shot'] = np.where(forward_mask, 1, 0)
    # turnover = shots_df.last_turnover
    # same_end = shots_df.last_same_end
    # shots_df['turnover_in_shot_end'] = np.where(turnover & same_end, 1, 0)
    prior_events = ['FACEOFF', 'SHOT', 'MISS', 'BLOCK',
                    'GIVEAWAY', 'TAKEAWAY', 'HIT']
    new_booleans = ['prior_faceoff', 'prior_shot', 'prior_miss', 'prior_block',
                    'prior_giveaway', 'prior_takeaway', 'prior_hit']
    for new_col, prior in zip(new_booleans, prior_events):
        shots_df[new_col] = np.where(shots_df.last_event_type == prior, 1, 0)
    # for prior_evt in prior_events:
    #     new_col = 'prior_' + prior_evt.lower()
    #     shots_df[new_col] = np.where(shots_df.last_event_type == prior_evt, 1, 0)
    # shots_df['prior_faceoff'] = np.where(shots_df.last_event_type == 'FACEOFF', 1, 0)
    # shots_df['prior_shot'] = np.where(shots_df.last_event_type == 'SHOT', 1, 0)
    # shots_df['prior_miss'] = np.where(shots_df.last_event_type == 'MISS', 1, 0)
    # shots_df['prior_block'] = np.where(shots_df.last_event_type == 'BLOCK', 1, 0)
    # shots_df['prior_give'] = np.where(shots_df.last_event_type == 'GIVEAWAY', 1, 0)
    # shots_df['prior_take'] = np.where(shots_df.last_event_type == 'TAKEAWAY', 1, 0)
    # shots_df['prior_hit'] = np.where(shots_df.last_event_type == 'HIT', 1, 0)

    return shots_df, ['forward_shot'] + new_booleans


def convert_bools_to_int(shots_df, columns):
    """ Converts boolean columns to integers (True -> 1, False -> 0).

    Parameters
        shots_df: dataframe = all shot data
        columns: list = columns to convert

    Returns
        shots_df: dataframe = updated shot data
    """
    for col in columns:
        shots_df[col] = shots_df[col].astype(int)

    return shots_df
