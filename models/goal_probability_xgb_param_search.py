import os
import pickle
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV as RSCV
from xgboost import XGBClassifier
from models.common_sql import create_db_connection, select_table
from models.common import sort_game_states


# TODO: calculate the probability of a block for different player positions, net
#  distances,
# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Create the database connection
connection = create_db_connection('postgres', 'marshineer', 'localhost', '5432',
                                  'password')

# Query the SQL database for shot data
shots_df = select_table(connection, 'shots')

# Sort the shot data
shots_df.sort_values(['game_id', 'shot_id'], inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Load the player data as a dictionary
players_df = select_table(connection, 'players')
players_list = players_df.to_dict('records')
players = {player_x['player_id']: player_x for player_x in players_list}

# Remove rows where the shift data contained errors
too_few_df = shots_df[shots_df.players_home.apply(lambda x: len(eval(x)) < 4) &
                      shots_df.players_away.apply(lambda x: len(eval(x)) < 4)]
too_many_df = shots_df[shots_df.players_home.apply(lambda x: len(eval(x)) > 6) |
                       shots_df.players_away.apply(lambda x: len(eval(x)) > 6)]
n_player_err = len(too_few_df) + len(too_many_df)
print(f'Number of shift error entries = {n_player_err} '
      f'({100 * n_player_err / len(shots_df):4.2f}% of all shots)')
shots_df = shots_df[shots_df.players_home.apply(lambda x: len(eval(x)) >= 4) &
                    shots_df.players_away.apply(lambda x: len(eval(x)) >= 4) &
                    shots_df.players_home.apply(lambda x: len(eval(x)) <= 6) &
                    shots_df.players_away.apply(lambda x: len(eval(x)) <= 6)]

# TODO: in ref9... set these to an average p(goal) value
# Remove rows with nan/null shot distance or angle
n_nan = len(shots_df[shots_df.net_distance.isna() | shots_df.net_angle.isna()])
print(f'Number of shots with no distance info = {n_nan} '
      f'({100 * n_nan / len(shots_df):4.2f}% of all shots)')
shots_df.dropna(subset=['net_distance', 'net_angle'], inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# TODO: in ref9... set these to an average p(goal) value for the given location
# Remove rows with nan/null previous shot coordinates
null_prev_mask = (shots_df.angle_change.isna() | shots_df.delta_x.isna() |
                  shots_df.delta_y.isna())
print(f'Number of shots with no previous event location info = '
      f'{sum(null_prev_mask)} ({100 * sum(null_prev_mask) / len(shots_df):4.2f}'
      f'% of all shots)')
shots_df.dropna(subset=['angle_change', 'delta_x', 'delta_y'], inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Remove blocks
block_mask = shots_df.shot_result == 'BLOCK'
blocks_df = shots_df.loc[block_mask]
blocks_df.reset_index(drop=True, inplace=True)
block_list = blocks_df.to_dict('records')
thru_prob = 1 - (sum(block_mask) / len(shots_df))
print(f'Number of shots that are blocked = '
      f'{len(block_mask)} ({100 * (1 - thru_prob):4.2f}% of all shots)')
shots_df.drop(shots_df.loc[block_mask].index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# TODO: in ref9... set p(goal) to 0 (should not be high anyway)
# Remove all shots that come from outside the blueline
long_mask = (((shots_df.period % 2 == 1) ^ (shots_df.shooter_home == True)) &
             (shots_df.x_coord < 25)) | \
            (~((shots_df.period % 2 == 1) ^ (shots_df.shooter_home == True)) &
             (shots_df.x_coord > -25))
print(f'Number of shots from outside the blueline = {sum(long_mask)} '
      f'({100 * sum(long_mask) / len(shots_df):4.2f}% of all shots)')
shots_df.drop(shots_df.loc[long_mask].index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Remove shots made by goalies
shots_df.drop(shots_df.loc[shots_df.shooter_position == 'G'].index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# TODO: in ref9... set p(goal) to 0 (impossible to shoot from here)
# Remove shots from directly behind the net
# The net is 72" wide and 40" deep, angle center to back corner = 138 degrees
behind_net_mask = (shots_df.net_angle > 138) | (shots_df.net_angle < -138)
print(f'Number of shots from directly behind the net = {sum(behind_net_mask)} '
      f'({100 * sum(behind_net_mask) / len(shots_df):4.2f}% of all shots)')
shots_df.drop(shots_df.loc[behind_net_mask].index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Covert boolean columns to numerical
shots_df.rebound_shot = shots_df.rebound_shot.astype(int)
shots_df.shooter_home = shots_df.shooter_home.astype(int)
shots_df.off_wing_shot = shots_df.off_wing_shot.astype(int)
shots_df.last_same_end = shots_df.last_same_end.astype(int)
shots_df.last_same_team = shots_df.last_same_team.astype(int)
shots_df.last_turnover = shots_df.last_turnover.astype(int)
shots_df.loc[shots_df.shooter_hand == 'L', 'shooter_hand'] = 0
shots_df.loc[shots_df.shooter_hand == 'R', 'shooter_hand'] = 1
shots_df.loc[shots_df.shooter_position == 'D', 'shooter_position'] = 0
shots_df.loc[shots_df.shooter_position == 'F', 'shooter_position'] = 1
shots_df.goal = shots_df.goal.astype(int)

# TODO: update everything to reflect the combined model with different models for
#  the game states: even, PP, PK, pulled goalie (with booleans or n_player feats)
# TODO: change this to sort into: even, PP, PK, empty net, and other states
# Divide the shots into their game states
game_states, state_lbls = sort_game_states(shots_df.to_dict('records'))

# Select a subset of features
feat_cols = ['net_distance', 'net_angle', 'delta_x', 'delta_y', 'angle_change',
             'time_since_last', 'goal_lead_prior', 'shot_time', 'shooter_age',
             'rebound_shot', 'shooter_home', 'shooter_hand', 'shooter_position',
             'off_wing_shot', 'last_same_end', 'last_same_team', 'last_turnover',
             'goal']

# Train one XGBoost model to determine best hyperparameters
cv_split = StratifiedKFold(n_splits=16)
xgb_params = []
xgb_score = []
n_params = 10
n_itr = 1000
for i, (state_ls, lbl) in enumerate(zip(game_states, state_lbls)):
    # Covert the shot list to a dataframe
    state_df = pd.DataFrame(state_ls)
    # state_df.drop(columns=drop_cols, inplace=True)
    # state_df = state_df[new_cols]
    state_df = state_df[feat_cols]
    print(f'\nThe {lbl} state contains {len(state_df)} shots')

    t_start = time()
    if i < 3:
        # Split the data into stratified training and test sets
        game_states[i] = state_df.to_dict('records')
        X_pd, y_pd = state_df.iloc[:, :-1].values, state_df.iloc[:, -1].values
        tmp_split = train_test_split(X_pd, y_pd, test_size=0.25, random_state=66,
                                     stratify=y_pd)
        X_train, X_test, y_train, y_test = tmp_split

        # Scale the numerical data (first 9 feature columns are numerical)
        n_num_cols = 9
        x_scaler = MinMaxScaler()
        X_train_norm = x_scaler.fit_transform(X_train[:, :n_num_cols])
        X_train[:, :n_num_cols] = X_train_norm
        X_test_norm = x_scaler.transform(X_test[:, :n_num_cols])
        X_test[:, :n_num_cols] = X_test_norm

        # Train the XGBoost model using a random parameter search
        # model = d_xgb.copy()
        model = {'name': 'XGBoost Hyperparameter Selection',
                 'model': XGBClassifier(use_label_encoder=False,
                                        tree_method='hist'),
                 'params': {'n_estimators': [100, 200, 300, 400, 600, 800, 1000],
                            'learning_rate': np.linspace(0.1, 0.5, 5),
                            'subsample': np.linspace(0.5, 1., 6),
                            'min_child_weight': [1, 5, 10, 20],
                            'max_leaves': [5, 10, 20, 40, 80],
                            'max_depth': np.arange(2, 9),
                            'gamma': [1, 2, 4, 8]}}
        param_search = RSCV(model['model'], model['params'], n_iter=n_itr,
                            scoring='neg_log_loss', cv=cv_split, n_jobs=-1)
        param_search.fit(X_train, y_train)
        search_results = pd.DataFrame(param_search.cv_results_)
        search_results.sort_values('rank_test_score', inplace=True)
        avg_loss = np.mean(search_results.mean_test_score)
        std_loss = np.std(search_results.mean_test_score)
        print(f'The average training loss is {avg_loss:4.3f} +/-{std_loss:4.3f}')
        print(f'The best {n_params} training loss scores are:')
        print(search_results.mean_test_score.iloc[:n_params])
        xgb_params.append(search_results.params.iloc[:n_params].tolist())
        xgb_score.append(search_results.mean_test_score.iloc[:n_params].tolist())

        t_end = time() - t_start
        print(f'Took {timedelta(seconds=t_end)} to train the {lbl} XGB model')

# Save the data required to train the models
pre_model_data = {'game_states': game_states,
                  'state_lbls': state_lbls,
                  'xgb_params': xgb_params,
                  'xgb_scores': xgb_score,
                  'features': feat_cols}

# Save the models
fpath = froot + f'/../data/goal_prob_model/pre_model_data_{n_itr}itr.pkl'
with open(fpath, 'wb') as f:
    pickle.dump(pre_model_data, f)
