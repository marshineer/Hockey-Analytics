import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
from torch import Tensor, sigmoid
from common_sql import create_db_connection, select_table
from models.common_plot import plot_in_game_probs, plot_calibration_curves


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Create the database connection
connection = create_db_connection('postgres', 'marshineer', 'localhost', '5432',
                                  'password')

# Load the game and shot data
print('Loading game data')
games_df = select_table(connection, 'games')
games_list = games_df.to_dict('records')
games = {game_x['game_id']: game_x for game_x in games_list}

print('Loading shot data')
shots_df = select_table(connection, 'shots')
shots_df = shots_df.loc[shots_df.shot_result.isin(['GOAL', 'SHOT'])]
# print(shots_df.columns.tolist())

# Extract the test games
# https://www.nhl.com/gamecenter/sea-vs-ari/2021/11/06/2021020172#game=2021020172,game_state=final
# https://www.nhl.com/gamecenter/nyr-vs-edm/2021/11/05/2021020158#game=2021020158,game_state=final
# https://www.nhl.com/gamecenter/nsh-vs-tbl/2021/01/30/2020020129#game=2020020129,game_state=final
# https://www.nhl.com/gamecenter/ana-vs-phx/2010/11/27/2010020341#game=2010020341,game_state=final
test_game_id1 = 2010020341
test_game_shots1 = shots_df.loc[shots_df.game_id == test_game_id1]
test_game_id2 = 2021020172
test_game_shots2 = shots_df.loc[shots_df.game_id == test_game_id2]
test_ids = [test_game_id1, test_game_id2]

# Load the model ensemble
t0_start = time()
print('Loading model ensemble')
fpath = '/../data/in_game_win_predictors/in_game_win_prediction_ensemble.pkl'
with open(froot + fpath, 'rb') as f:
    ens_models = pickle.load(f)
print(f'Took {timedelta(seconds=time() - t0_start)} to load the ensemble')

# Calculate the win probability across game times and goal differentials
ens_size = len(ens_models)
game_len = 3600
ens_ind = 0
goal_diffs = [2, 1, 0, -1]
model_probs = np.zeros((ens_size, len(goal_diffs), game_len))
for i, model in enumerate(ens_models):
    scaler = model['x_scaler']
    input_arr = np.zeros((game_len, 3))
    for j, g_diff in enumerate(goal_diffs):
        input_arr[:, 0] = g_diff
        input_arr[:, -1] = np.arange(game_len)[::-1]
        # input_arr[:, -1] = np.arange(game_len)
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
fig, _, _ = plot_calibration_curves(model_preds, y_tests, class1='Home Win',
                                    plt_ttl=plt_ttl)
fig.savefig(froot + f'/../readme_imgs/ensemble_calibration_curve.png')
