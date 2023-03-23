import os
import pickle
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV as RSCV
from xgboost import XGBClassifier
from models.common_sql import create_db_connection, select_table
from models.common import sort_game_states, create_stratify_feat, \
    normalize_continuous


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

# # Load the player data as a dictionary
# players_df = select_table(connection, 'players')
# players_list = players_df.to_dict('records')
# players = {player_x['player_id']: player_x for player_x in players_list}

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

# TODO: in ref9... set p(goal) to 0 (should not be high anyway)
# Remove all shots that come from outside the blueline
long_mask = (((shots_df.period % 2 == 1) ^ (shots_df.shooter_home == True)) &
             (shots_df.x_coord < 25)) | \
            (~((shots_df.period % 2 == 1) ^ (shots_df.shooter_home == True)) &
             (shots_df.x_coord > -25))
print(f'Number of shots from outside the blueline = {sum(long_mask)} '
      f'({100 * sum(long_mask) / len(shots_df):4.2f}% of all shots)')
shots_df['en_shot'] = shots_df.apply(lambda x: (x.shooter_home and
                                                x.empty_net_away) or
                                               (not x.shooter_home and
                                                x.empty_net_home),
                                     axis=1)
long_shots = shots_df.loc[long_mask & ~shots_df.en_shot]
print(f'{100 * len(long_shots.loc[long_shots.goal]) / len(long_shots):4.2f}% '
      f'of long shots result in a goal')
shots_df.drop(shots_df.loc[long_mask].index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Remove shots made by goalies
shots_df.drop(shots_df.loc[shots_df.shooter_position == 'G'].index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# TODO: in ref9... set p(goal) to 0 (impossible to shoot on net from here)
# Remove shots from directly behind the net
# The net is 72" wide and 40" deep, angle center to back corner = 138 degrees
behind_net_mask = (shots_df.net_angle > 138) | (shots_df.net_angle < -138)
print(f'Number of shots from directly behind the net = {sum(behind_net_mask)} '
      f'({100 * sum(behind_net_mask) / len(shots_df):4.2f}% of all shots)')
shots_df.drop(shots_df.loc[behind_net_mask].index, inplace=True)
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

# Additional features used in the expected goals model
forward_mask = shots_df.shooter_position == 'F'
shots_df['forward_shot'] = np.where(forward_mask, 1, 0)
turnover_mask = shots_df.last_turnover
same_end_mask = shots_df.last_same_end
shots_df['turnover_in_shot_end'] = np.where(turnover_mask & same_end_mask, 1, 0)
shots_df['prior_faceoff'] = np.where(shots_df.last_event_type == 'FACEOFF', 1, 0)
shots_df['prior_shot'] = np.where(shots_df.last_event_type == 'SHOT', 1, 0)
shots_df['prior_miss'] = np.where(shots_df.last_event_type == 'MISS', 1, 0)
shots_df['prior_block'] = np.where(shots_df.last_event_type == 'BLOCK', 1, 0)
shots_df['prior_give'] = np.where(shots_df.last_event_type == 'GIVEAWAY', 1, 0)
shots_df['prior_take'] = np.where(shots_df.last_event_type == 'TAKEAWAY', 1, 0)
shots_df['prior_hit'] = np.where(shots_df.last_event_type == 'HIT', 1, 0)

# Covert existing boolean columns to numerical (integer)
shots_df.rebound_shot = shots_df.rebound_shot.astype(int)
shots_df.shooter_home = shots_df.shooter_home.astype(int)
shots_df.off_wing_shot = shots_df.off_wing_shot.astype(int)
# shots_df.last_same_end = shots_df.last_same_end.astype(int)
shots_df.last_same_team = shots_df.last_same_team.astype(int)
shots_df.pulled_goalie = shots_df.pulled_goalie.astype(int)
# shots_df.last_turnover = shots_df.last_turnover.astype(int)
# shots_df.loc[shots_df.shooter_hand == 'L', 'shooter_hand'] = 0
# shots_df.loc[shots_df.shooter_hand == 'R', 'shooter_hand'] = 1
# shots_df.loc[shots_df.shooter_position == 'D', 'shooter_position'] = 0
# shots_df.loc[shots_df.shooter_position == 'F', 'shooter_position'] = 1
shots_df.goal = shots_df.goal.astype(int)

# Select a subset of features
cont_feats = ['net_distance',  'net_angle', 'dist_change', 'angle_change',
              'shot_time', 'time_since_last', 'goal_lead_prior']
bool_feats = ['shooter_home', 'forward_shot', 'off_wing_shot', 'rebound_shot',
              'last_same_team', 'turnover_in_shot_end', 'pulled_goalie',
              'prior_faceoff', 'prior_shot', 'prior_miss', 'prior_block',
              'prior_give', 'prior_take', 'prior_hit']
train_features = cont_feats + bool_feats

# Normalize the continuous features used in training
shots_df_norm = normalize_continuous(shots_df, cont_feats)[0]

# TODO: update everything to reflect the combined model with different models for
#  the game states: even, PP, PK, pulled goalie (with booleans or n_player feats)
# TODO: change this to sort into: even, PP, PK, empty net, and other states
# Divide the shots into their game states
game_states, state_lbls = sort_game_states(shots_df_norm.to_dict('records'))

# Define the XGBoost hyperparameter ranges
# https://xgboost.readthedocs.io/en/stable/python/python_api.html
# xgb_params = {'n_estimators': [50, 100, 200, 400, 800],
#               'learning_rate': np.linspace(0.01, 0.9, 90),
#               'subsample': np.linspace(0.2, 1., 9),
#               'min_child_weight': [1, 5, 10, 20, 40],
#               'max_leaves': [2, 5, 10, 20, 40, 80],
#               'max_depth': np.arange(2, 12),
#               'lambda': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
#               'gamma': [1, 2, 4, 8, 12, 16, 24]}
# Refinements based on analysis in "my_xG_model_refine_hyperparameters.ipynb"
xgb_params = {'n_estimators': np.arange(100, 401, 25),
              'learning_rate': np.linspace(0.01, 0.3, 30),
              'subsample': np.linspace(0.7, 1., 31),
              'min_child_weight': np.linspace(1, 10, 91),
              'max_leaves': np.arange(20, 101, 10),
              'max_depth': np.arange(3, 10),
              'gamma': np.linspace(0.01, 0.5, 50)}

# Train one XGBoost model to determine the best hyperparameters
cv_params = {}
cv_scores = {}
strength_n_goals = {}
strength_n_shots = {}
strength_p_goal = {}
strength_dumb_loss = {}
state_dumb_loss = {}
strength_lbls = {}
n_params = 10
n_itr = 1000
seed = 66
for i, (state_shots, state_lbl) in enumerate(zip(game_states, state_lbls)):
    # Omit irrelevant states
    if state_lbl not in ['Even', 'PP', 'PK']:
        print(f'The state {state_lbl} is not modelled')
        continue

    # Covert the shot list to a dataframe
    state_df = pd.DataFrame(state_shots, columns=list(state_shots[0].keys()))
    print(f'\nThe {state_lbl} state contains {len(state_df)} shots')

    # Define the parameters for each game strength
    strengths = None
    n_folds = None
    if state_lbl == 'Even':
        strengths = ['5v5', '4v4', '3v3']
        n_folds = 20
    elif state_lbl == 'PP':
        strengths = ['5v4', '5v3', '4v3']
        n_folds = 20
    elif state_lbl == 'PK':
        strengths = ['4v5', '3v5', '3v4']
        n_folds = 5
    strength_lbls[state_lbl] = strengths

    # Calculate the goal probability and dumb loss for the given states/strengths
    n_goals = len(state_df.loc[state_df.goal == True])
    n_shots = len(state_df)
    p_goal = n_goals / n_shots
    dumb_loss = -(p_goal * np.log(p_goal) + (1 - p_goal) * np.log(1 - p_goal))
    state_dumb_loss[state_lbl] = dumb_loss
    for strength in strengths:
        strength_df = state_df.loc[state_df[strength] == 1]
        n_goals = len(strength_df.loc[strength_df.goal == True])
        n_shots = len(strength_df)
        p_goal = n_goals / n_shots
        dumb_loss = -(p_goal * np.log(p_goal) +
                      (1 - p_goal) * np.log(1 - p_goal))
        strength_n_goals[strength] = n_goals
        strength_n_shots[strength] = n_shots
        strength_p_goal[strength] = p_goal
        strength_dumb_loss[strength] = dumb_loss

    # Define a column to stratify the data by game state and target value
    # https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
    # https://stackoverflow.com/questions/30040597/how-to-generate-a-custom-cross-validation-generator-in-scikit-learn
    # https://medium.com/@antoniolui/applying-custom-functions-in-pandas-e30bdc1f4e76
    strat_cols = strengths + ['goal']
    state_df['stratify'] = state_df.apply(lambda x:
                                          create_stratify_feat(x[strat_cols]),
                                          axis=1)

    # Split the data into features and target variable
    bool_cols = bool_feats + strengths
    X_pd, y_pd = state_df[train_features].values, state_df.goal.values

    # Define a stratified cross-validation split
    stratify_feature = state_df.stratify.values
    cv_split = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    cv_generator = cv_split.split(X_pd, stratify_feature)

    # Initialize the model
    xgb_model = XGBClassifier(use_label_encoder=False, tree_method='hist',
                              random_state=seed)

    # Train the model
    t_start = time()
    param_search = RSCV(xgb_model, xgb_params, n_iter=n_itr,
                        scoring='neg_log_loss', n_jobs=-1, cv=cv_generator)
    param_search.fit(X_pd, y_pd)
    cv_results = pd.DataFrame(param_search.cv_results_)
    cv_results.sort_values('rank_test_score', inplace=True)
    cv_results.dropna(axis=0, inplace=True)
    avg_loss = np.mean(cv_results.mean_test_score)
    std_loss = np.std(cv_results.mean_test_score)
    print(f'The average training loss is {avg_loss:4.3f} +/-{std_loss:4.3f}')
    print(f'The best {n_params} training loss scores are:')
    print(cv_results.mean_test_score.iloc[:n_params])
    cv_params[state_lbl] = cv_results.params.tolist()
    cv_scores[state_lbl] = cv_results.mean_test_score.tolist()

    # Print training time
    t_end = time() - t_start
    print(f'Took {timedelta(seconds=t_end)} to train the {state_lbl} XGB model')

# Save the data required to train the models
pre_model_data = {'game_states': game_states,
                  'state_lbls': state_lbls,
                  'random_state': seed,
                  'xgb_params': xgb_params,
                  'cv_params': cv_params,
                  'cv_scores': cv_scores}

state_strength_info = {'continuous_features': cont_feats,
                       'boolean_features': bool_feats,
                       'state_lbls': state_lbls,
                       'state_dumb_loss': strength_dumb_loss,
                       'strength_lbls': strength_lbls,
                       'strength_n_goals': strength_n_goals,
                       'strength_n_shots': strength_n_shots,
                       'strength_p_goal': strength_p_goal,
                       'strength_dumb_loss': strength_dumb_loss}

# Save the hyperparameter data
fpath = froot + f'/../data/goal_prob_model/hyperparameter_search_{n_itr}itr.pkl'
with open(fpath, 'wb') as f:
    pickle.dump(pre_model_data, f)

# Save the state data
fpath = froot + f'/../data/goal_prob_model/state_strength_info.pkl'
with open(fpath, 'wb') as f:
    pickle.dump(state_strength_info, f)
