import os
import pickle
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch import optim, Tensor, sigmoid, manual_seed
from torch.utils.data import DataLoader
from models.common_sql import create_db_connection, select_table
from models.common_torch import RegressionNN, CustomDataset, train_loop


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Create the database connection
connection = create_db_connection('postgres', 'marshineer', 'localhost', '5432',
                                  'password')

# Load the game and shot data
games_df = select_table(connection, 'games')
games_list = games_df.to_dict('records')
games = {game_x['game_id']: game_x for game_x in games_list}

shots_df = select_table(connection, 'shots')
shots_df = shots_df.loc[shots_df.shot_result.isin(['GOAL', 'SHOT'])]

# Remove select data from dataset
print('Removed:')

# # Remove empty net goals
# en_goals = shots_df[shots_df.empty_net == True]
# frac_en_goals = len(en_goals) / len(shots_df[shots_df.shot_result == "GOAL"])
# print(f'Number of goals on empty nets = {len(en_goals)} '
#       f'({100 * frac_en_goals:4.2f}% of all goals)')
# shots_df.drop(en_goals.index, inplace=True)
# shots_df.reset_index(drop=True, inplace=True)

# Remove overtime games
overtime_games = games_df.loc[games_df.number_periods > 3]
print(f'Number of overtime games = {len(overtime_games)} '
      f'({100 * len(overtime_games) / len(games_df):4.2f}% of all games)')
shots_df.drop(shots_df[shots_df.game_id.isin(overtime_games.game_id.tolist())].
              index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Remove playoff games
playoff_games = games_df.loc[games_df.type == 'PLA']
print(f'Number of playoff games = {len(playoff_games)} '
      f'({100 * len(playoff_games) / len(games_df):4.2f}% of all games)')
playoff_df = shots_df[shots_df.game_id.isin(playoff_games.game_id.tolist())]
shots_df.drop(playoff_df.index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Remove the test games
# https://www.nhl.com/gamecenter/sea-vs-ari/2021/11/06/2021020172#game=2021020172,game_state=final
# https://www.nhl.com/gamecenter/nyr-vs-edm/2021/11/05/2021020158#game=2021020158,game_state=final
# https://www.nhl.com/gamecenter/nsh-vs-tbl/2021/01/30/2020020129#game=2020020129,game_state=final
# https://www.nhl.com/gamecenter/ana-vs-phx/2010/11/27/2010020341#game=2010020341,game_state=final
test_game_id1 = 2010020341
test_game_shots1 = shots_df.loc[shots_df.game_id == test_game_id1]
test_game_id2 = 2021020172
test_game_shots2 = shots_df.loc[shots_df.game_id == test_game_id2]
test_ids = [test_game_id1, test_game_id2]
shots_df.drop(shots_df[shots_df.game_id.isin(test_ids)].index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Group the shots by game ID
shots_gb = shots_df.groupby('game_id')

# Define the input features and target variable
data_cols = ['goal_diff', 'shot_diff', 'game_time', 'home_win']
game_len = 3 * 20 * 60
game_id_list = shots_df.game_id.unique().tolist()
n_games = len(game_id_list)
n_samples = 150
input_arr = np.zeros((n_samples * n_games, len(data_cols)))
for i, game_id in enumerate(game_id_list):
    # Set the goal and shot differential data
    goal_diff = 0
    shot_diff = 0
    prev_time = 0
    shots_list = shots_gb.get_group(int(game_id)).to_dict('records')

    # Initialize the second array
    game_sec_inputs = np.zeros((game_len, len(data_cols)))
    game_sec_inputs[:, 2] = np.arange(game_len)[::-1]
    # game_sec_inputs[:, -2] = np.arange(game_len)
    game_sec_inputs[:, -1] = 1 if games[game_id]['home_win'] else 0
    for j, shot in enumerate(shots_list):
        # Record the data between the previous and current shots
        this_time = shot['shot_time']
        game_sec_inputs[prev_time:this_time, 0] = goal_diff
        game_sec_inputs[prev_time:this_time, 1] = shot_diff
        # if j < (len(shots_list) - 1):
        #     game_sec_inputs[prev_time:this_time, 0] = goal_diff
        #     game_sec_inputs[prev_time:this_time, 1] = shot_diff
        # else:
        #     game_sec_inputs[prev_time:, 0] = goal_diff
        #     game_sec_inputs[prev_time:, 1] = shot_diff
        prev_time = this_time

        # Update shot differential
        if shot['shooter_home']:
            shot_diff += 1
        else:
            shot_diff -= 1

        # Update goal differential
        goal_diff = shot['home_score'] - shot['away_score']

    # Add data after last shot
    game_sec_inputs[prev_time:, 0] = goal_diff
    game_sec_inputs[prev_time:, 1] = shot_diff

    # Subsample the game
    sample_inds = np.random.choice(game_len, size=n_samples, replace=False)
    game_samples = game_sec_inputs[sample_inds, :]
    input_arr[i * n_samples:(i + 1) * n_samples, :] = game_samples

data_df = pd.DataFrame(input_arr, columns=data_cols)

# Define the ensemble seeds
# print(np.random.choice(np.arange(1e4, 1e5).astype(int), size=50, replace=False))
# ens_seeds = [99669, 41975, 77840, 92597, 30427, 86846, 54781, 72793, 46298,
#              26165, 99995, 10038, 37807, 52924, 99469, 49268, 40677, 41554,
#              74175, 52253, 22453, 29474, 65014, 40201, 81510, 15734, 48159,
#              38745, 45299, 70848, 12202, 38238, 21620, 82789, 38227, 41272,
#              10766, 78230, 92645, 57404, 13953, 51528, 77956, 16312, 39888,
#              14233, 94609, 18560, 37869, 42528]
ens_seeds = [99669, 41975, 77840, 92597, 30427, 86846, 54781, 72793, 46298,
             26165, 99995, 10038, 37807, 52924, 99469, 49268, 40677, 41554,
             74175, 52253, 38737, 29474, 65014, 40201, 81510, 15734, 48159,
             38745, 45299, 70848, 12202, 38238, 21620, 82789, 38227, 41272,
             10766, 78230, 92645, 57404, 13953, 51528, 77956, 16312, 39888,
             14233, 94609, 18560, 37869, 42528, 16774, 46723, 99723, 23830,
             88686, 34055, 90634, 46324, 62138, 98342, 85639, 21804, 46297,
             69679, 65987, 66319, 25344, 45295, 45762, 85837, 89508, 90239,
             93787, 38709, 58878, 76755, 38157, 51337, 28136, 62995, 15447,
             70171, 74787, 62355, 18063, 28649, 26921, 48649, 69363, 70930,
             24827, 13452, 93889, 51836, 26267, 24095, 39206, 27959, 75098,
             54366]

# Define hyperparameters
ens_size = len(ens_seeds)
learning_rate = 1e-4
batch_size = 512
n_epochs = 25
loss_fn = nn.BCEWithLogitsLoss()

ens_models = []
train_loss_avg = np.zeros((ens_size, n_epochs))
train_loss_std = np.zeros((ens_size, n_epochs))
t0_start = time()
for i, seed in enumerate(ens_seeds):
    # Set the pytorch and numpy seeds
    # https://pytorch.org/docs/stable/notes/randomness.html
    manual_seed(seed)
    np.random.seed(seed)

    # Split the data into training and test sets
    X_pd, y_pd = data_df.iloc[:, :-1].values, data_df.iloc[:, -1].values
    tmp_split = train_test_split(X_pd, y_pd, test_size=0.25, random_state=seed)
    X_train, X_test, y_train, y_test = tmp_split

    # Scale the data
    x_scaler = MinMaxScaler()
    X_train_norm = x_scaler.fit_transform(X_train)
    X_test_norm = x_scaler.transform(X_test)

    # Define the training and test data loaders
    train_data = CustomDataset(X_train_norm, y_train)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)

    # Initialize a model and optimizer
    model = RegressionNN(X_train_norm.shape[-1], 32, 1).to('cpu').float()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the ensemble
    t_start = time()
    for t in range(n_epochs):
        epoch_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        train_loss_avg[i, t] = np.mean(epoch_loss)
        train_loss_std[i, t] = np.std(epoch_loss)

    # Print the training time taken for one model
    print(f'Took {timedelta(seconds=time() - t_start)} to train model #{i + 1}')

    # Test the model
    train_pred = sigmoid(model(Tensor(X_train_norm))).detach().numpy().squeeze()
    train_loss = log_loss(y_train, train_pred, eps=1e-7)
    test_pred = sigmoid(model(Tensor(X_test_norm))).detach().numpy().squeeze()
    test_loss = log_loss(y_test, test_pred, eps=1e-7)
    pred_classes = np.round(test_pred, 0)
    test_acc = 100 * (pred_classes == y_test).sum() / y_test.size

    # Save model as a dictionary
    d_model = {'model': model,
               'seed': seed,
               # 'x_train': X_train,
               # 'x_test': X_test,
               'x_scaler': x_scaler,
               'y_train': y_train,
               'y_test': y_test,
               'y_pred_train': train_pred,
               'y_pred_test': test_pred,
               'train_loss': train_loss,
               'test_loss': test_loss,
               'test_acc': test_acc}
    ens_models.append(d_model)

# Print the time taken
print(f'Took {timedelta(seconds=time() - t0_start)} to train the whole ensemble')

# Save the models_and_analysis
# with open(froot + '/../data/in_game_win_prediction_ensemble.pkl', 'wb') as f:
#     pickle.dump(ens_models, f)
fpath = '/../data/in_game_win_prediction_ensemble_reverse.pkl'
with open(froot + fpath, 'wb') as f:
    pickle.dump(ens_models, f)
