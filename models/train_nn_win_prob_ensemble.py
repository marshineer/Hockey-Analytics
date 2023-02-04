import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch import optim, Tensor, sigmoid, manual_seed
from torch.utils.data import DataLoader
from common_sql import create_db_connection, select_table
from nhl_api.ref_common import game_time_to_sec
from models.common_torch import RegressionNN, CustomDataset, train_loop
from models.common_plot import plot_in_game_probs, plot_calibration_curves


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
# print(shots_df.columns.tolist())

# Remove select data from dataset
print('Removed:')

# Remove empty net goals
en_goals = shots_df[shots_df.empty_net == True]
frac_en_goals = len(en_goals) / len(shots_df[shots_df.shot_result == "GOAL"])
print(f'Number of goals on empty nets = {len(en_goals)} '
      f'({100 * frac_en_goals:4.2f}% of all goals)')
shots_df.drop(en_goals.index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# Remove overtime games
overtime_games = games_df.loc[games_df.number_periods > 3]
print(f'Number of overtime games = {len(overtime_games)} '
      f'({100 * len(overtime_games) / len(games_df):4.2f}% of all games)')
shots_df.drop(shots_df[shots_df.game_id.isin(overtime_games.game_id.tolist())].
              index, inplace=True)
shots_df.reset_index(drop=True, inplace=True)

# # Remove overtime shots
# print(f'Number of shots in overtime = {len(shots_df[shots_df.period > 3])} '
#       f'({100 * len(shots_df[shots_df.period > 3]) / len(shots_df):4.2f}% of all shots)')
# shots_df.drop(shots_df[shots_df.period > 3].index, inplace=True)
# shots_df.reset_index(drop=True, inplace=True)

# # Remove shootout games
# shootout_games = games_df.loc[games_df.shootout]
# print(f'Number of shootout games = {len(shootout_games)} '
#       f'({100 * len(shootout_games) / len(games_df):4.2f}% of all games)')
# shots_df.drop(shots_df[shots_df.game_id.isin(shootout_games.game_id.tolist())].index, inplace=True)
# shots_df.reset_index(drop=True, inplace=True)

# Remove playoff games
playoff_games = games_df.loc[games_df.type == 'PLA']
print(f'Number of playoff games = {len(playoff_games)} '
      f'({100 * len(playoff_games) / len(games_df):4.2f}% of all games)')
shots_df.drop(shots_df[shots_df.game_id.isin(playoff_games.game_id.tolist())].index, inplace=True)
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

# Convert the shots to a list
shots_list = shots_df.to_dict('records')

# Define the input features and target variable
data_cols = ['goal_diff', 'shot_diff', 'game_time', 'home_win']
game_len = 20 * 3 * 60
this_game_id = None
home_shots = None
away_shots = None
home_win = None
data_list = []
data_list_gameid = []
# n_games = 0
# n_home_wins = 0
for shot in shots_list:
    game_id = shot['game_id']
    if this_game_id != game_id:
        # game_data = games[game_id]
        # game_shots = shots_df[shots_df.game_id == game_id].to_dict('records')
        # game_len = calc_game_length(game_shots, game_data)
        this_game_id = game_id
        home_win = 1 if games[game_id]['home_win'] else 0
        # if home_win == 1:
        #     n_home_wins += 1
        # n_games += 1
        if shot['shooter_home']:
            home_shots = 1
            away_shots = 0
        else:
            home_shots = 0
            away_shots = 1
    else:
        if shot['shooter_home']:
            home_shots += 1
        else:
            away_shots += 1
    period = shot['period']
    period_time = shot['period_time']
    game_time = (period - 1) * 20 * 60 + game_time_to_sec(period_time)
    game_time = game_len - game_time
    home_score = shot['home_score']
    away_score = shot['away_score']
    data_list.append([home_score - away_score, home_shots - away_shots,
                      game_time, home_win])

data_df = pd.DataFrame(data_list, columns=data_cols)

# Define the ensemble seeds
# print(np.random.choice(np.arange(1e4, 1e5).astype(int), size=25, replace=False))
ens_seeds = [99669, 41975, 77840, 92597, 93678, 86846, 86827, 72793, 46298,
             26165, 99995, 10038, 37807, 52924, 99469, 49268, 40677, 41554,
             74175, 58768, 55909, 29474, 65014, 40201, 81510, 15734, 48159,
             38745, 45299, 36448, 12202, 38238, 21620, 82789, 38227, 41272,
             10766, 78230, 92645, 57404, 13953, 51528, 77956, 16312, 39888,
             14233, 94609, 18560, 37869, 42528]

# Define hyperparameters
ens_size = len(ens_seeds)
learning_rate = 1e-4
batch_size = 512
n_epochs = 100
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
    y_pred_train = sigmoid(model(Tensor(X_train_norm))).detach().numpy().squeeze()
    train_loss = log_loss(y_train, y_pred_train, eps=1e-7)
    y_pred_test = sigmoid(model(Tensor(X_test_norm))).detach().numpy().squeeze()
    test_loss = log_loss(y_test, y_pred_test, eps=1e-7)
    pred_classes = np.round(y_pred_test, 0)
    test_acc = 100 * (pred_classes == y_test).sum() / y_test.size

    # Save model as a dictionary
    d_model = {'model': model,
               'seed': seed,
               'x_train': X_train,
               'x_test': X_test,
               'x_scaler': x_scaler,
               'y_train': y_train,
               'y_test': y_test,
               'y_pred_train': y_pred_train,
               'y_pred_test': y_pred_test,
               'train_loss': train_loss,
               'test_loss': test_loss,
               'test_acc': test_acc}
    ens_models.append(d_model)

# Print the time taken
print(f'Took {timedelta(seconds=time() - t0_start)} to train the whole ensemble')

# Save the models_and_analysis
with open(froot + '/../data/in_game_win_prediction_ensemble.pkl', 'wb') as f:
    pickle.dump(ens_models, f)

# Plot the prediction distribution
ens_ind = 0
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# ax.hist(ens_models[ens_ind]['y_pred_test'], bins=50)
for i in range(ens_size):
    # ax.hist(ens_models[i]['y_pred_test'], bins=50, histtype='step')
    ax.hist(ens_models[i]['y_pred_test'], bins=50, alpha=0.1)
ax.set_title('Neural Network Prediction Distribution', fontsize=16)
ax.set_xlabel("Probability of 'Home Win'", fontsize=12)
ax.set_ylabel('Count', fontsize=12)
# plt.show()

# Calculate the win probability across game times and goal differentials
goal_diffs = [2, 1, 0, -1]
model_probs = np.zeros((ens_size, len(goal_diffs), game_len))
for i, model in enumerate(ens_models):
    scaler = model['x_scaler']
    input_arr = np.zeros((game_len, 3))
    for j, g_diff in enumerate(goal_diffs):
        input_arr[:, 0] = g_diff
        input_arr[:, -1] = np.arange(game_len)[::-1]
        scaled_input = scaler.transform(input_arr)
        pred_prob = sigmoid(model['model'](Tensor(scaled_input)))
        model_probs[i, j, :] = pred_prob.detach().numpy().squeeze()

# Plot the continuous win probability for goal differentials of +2, +1, 0 and -1
line_stys = ['--', '-', '-.', ':']
plt_ttls = [f'Model #{ens_ind}', 'Ensemble Average']
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
for i, probs in enumerate([model_probs[ens_ind], np.mean(model_probs, axis=0)]):
    for j, goal_diff in enumerate(goal_diffs):
        lbl = f'+{goal_diff}' if goal_diff > 0 else goal_diff
        axes[i].plot(np.arange(game_len), probs[j, :], f'k{line_stys[j]}',
                     label=f'{lbl} Home Lead')
    axes[i].set_title(plt_ttls[i], fontsize=14)
    axes[i].set_ylabel('Home Win Probability', fontsize=12)
    axes[i].legend(loc='upper left', bbox_to_anchor=(1.0, 1.03))
    axes[i].set_ylim([0, 1])
axes[-1].set_xlabel('Game Time (seconds)', fontsize=12)
fig.tight_layout()
fig.savefig(froot + '/../readme_imgs/home_win_prob_vs_goal_diffs.png')

# Plot the in-game win probability for an example game
if test_game_id1 == 2021020172:
    team_names1 = ['Arizona', 'Seattle']
elif test_game_id1 == 2021020158:
    team_names1 = ['Edmonton', 'New York']
elif test_game_id1 == 2020020129:
    team_names1 = ['Tampa Bay', 'Nashville']
elif test_game_id1 == 2010020341:
    team_names1 = ['Arizona', 'Anaheim']
test_home_win1 = games[test_game_id1]['home_win']
x_scalers = [model['x_scaler'] for model in ens_models]
fig, _ = plot_in_game_probs(test_game_shots1, test_home_win1, team_names1,
                            x_scalers, ens_models)
fig.savefig(froot + f'/../readme_imgs/in_game_win_prob_{test_game_id1}.png')

# Plot the in-game win probability for another example game
if test_game_id2 == 2021020172:
    team_names2 = ['Arizona', 'Seattle']
elif test_game_id2 == 2021020158:
    team_names2 = ['Edmonton', 'New York']
elif test_game_id2 == 2020020129:
    team_names2 = ['Tampa Bay', 'Nashville']
elif test_game_id2 == 2010020341:
    team_names2 = ['Arizona', 'Anaheim']
test_home_win2 = games[test_game_id2]['home_win']
fig, _ = plot_in_game_probs(test_game_shots2, test_home_win2, team_names2,
                            x_scalers, ens_models)
fig.savefig(froot + f'/../readme_imgs/in_game_win_prob_{test_game_id2}.png')

# Plot the ensemble calibration curve
model_preds = [model['y_pred_test'] for model in ens_models]
y_tests = [model['y_test'] for model in ens_models]
plt_ttl = 'Ensemble Calibration Curve'
fig, _, _ = plot_calibration_curves(ens_models, model_preds, y_tests,
                                    class1='Home Win', plt_title=plt_ttl)
fig.savefig(froot + f'/../readme_imgs/ensemble_calibration_curve.png')
