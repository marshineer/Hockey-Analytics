import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from torch import Tensor, sigmoid
from common_sql import create_db_connection, select_table
from models.common_plot import plot_in_game_probs, plot_calibration_curves


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# # Create the database connection
# connection = create_db_connection('postgres', 'marshineer', 'localhost', '5432',
#                                   'password')

# Load the input feature names
info_path = f'/../data/goal_prob_model/expected_goal_modelling_info.pkl'
with open(froot + info_path, 'rb') as f:
    state_strength_info = pickle.load(f)
# cont_cols = state_strength_info['continuous_features']
# bool_cols = state_strength_info['boolean_features']
state_lbls = state_strength_info['state_lbls']
strength_lbls = state_strength_info['strength_lbls']
# https://stackoverflow.com/questions/13016129/python-flatten-a-dict-of-lists-into-unique-values
# game_strength_list = [x for v in strength_lbls.values() for x in v]
all_dumb_losses = state_strength_info['strength_dumb_loss']
game_strength_dumb_loss = [loss for loss in all_dumb_losses.values()]
# game_state_dumb_loss = state_strength_info['state_dumb_loss']
# state_lbls = ['Even', 'PP', 'PK']
# n_states = len(state_lbls)

# Load the model ensemble
n_itr = 100
fpath = f'/../data/goal_prob_model/goal_prob_game_state_ensembles_{n_itr}itr.pkl'
with open(froot + fpath, 'rb') as f:
    game_state_models = pickle.load(f)

# Calculate the values for plotting
game_strength_losses = []
game_strength_preds = []
game_strength_ys = []
game_strength_list = []
for state_lbl in state_lbls:
    model_list = game_state_models[state_lbl]
    strength_cols = strength_lbls[state_lbl]
    for strength in strength_cols:
        game_strength_list.append(strength)
        ys_list = []
        preds_list = []
        loss_list = []
        for model in model_list:
            strength_inds = model['strength_test_inds'][strength]
            ys_list.append(model['y_test'][strength_inds])
            preds_list.append(model['test_pred'][strength_inds])
            loss_list.append(log_loss(ys_list[-1], preds_list[-1]))
        ys_arr = np.hstack(ys_list)
        preds_arr = np.hstack(preds_list)
        # game_strength_losses.append(log_loss(ys_arr, preds_arr))
        game_strength_losses.append(loss_list)
        game_strength_ys.append(ys_arr)
        game_strength_preds.append(preds_arr)

# # Calculate the values for plotting
# game_state_losses = []
# game_state_preds = []
# game_state_ys = []
# for state_lbl in state_lbls:
#     model_list = game_state_models[state_lbl]
#     game_state_losses.append([model['test_loss'] for model in model_list])
#     pred_list = [model['test_pred'] for model in model_list]
#     y_list = [model['y_test'] for model in model_list]
#     game_state_preds.append(np.hstack(pred_list))
#     game_state_ys.append(np.hstack(y_list))

# Plot the training loss and trivial loss for each state
max_dumb = max(game_strength_dumb_loss)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
xs = np.arange(len(game_strength_dumb_loss)) + 0.5
ax.scatter(xs, game_strength_dumb_loss, c='k', marker='*', label='Dumb Loss')
ax.violinplot(game_strength_losses, xs, showmeans=True)
# xs = np.arange(len(game_state_dumb_loss)) + 0.5
# ax.scatter(xs, game_state_dumb_loss.values(), c='k', marker='*',
#            label='Dumb Loss')
# ax.violinplot(game_state_losses, xs, showmeans=True)
ax.set_xlabel('Game State', fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels(game_strength_list, fontsize=10)
# ax.set_xticklabels(state_lbls, fontsize=10)
ax.set_ylabel('Logloss', fontsize=12)
ax.set_ylim([0, 1.1 * max_dumb])
ax.legend(loc=2)
fig.savefig(froot + f'/../readme_imgs/goal_prediction_loss_violin.png')

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(xs, game_strength_dumb_loss, c='k', marker='*', label='Dumb Loss')
ax.boxplot(game_strength_losses, positions=xs, showmeans=True)
# ax.scatter(xs, game_state_dumb_loss.values(), c='k', marker='*',
#            label='Dumb Loss')
# ax.boxplot(game_state_losses, positions=xs, showmeans=True)
ax.set_xlabel('Game State', fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels(game_strength_list, fontsize=10)
# ax.set_xticklabels(state_lbls, fontsize=10)
ax.set_ylabel('Logloss', fontsize=12)
ax.set_ylim([0, 1.1 * max_dumb])
ax.legend(loc=2)
fig.savefig(froot + f'/../readme_imgs/goal_prediction_loss_box.png')

# Plot the calibration curves
cal_ttl = 'Expected Goal Calibration Curves'
fig, _, _ = plot_calibration_curves(game_strength_preds, game_strength_ys,
                                    n_bins=20, names=game_strength_list,
                                    avg_curve=False, density=True,
                                    class1='Goal', plt_ttl=cal_ttl)
# fig, _, _ = plot_calibration_curves(game_state_preds, game_state_ys,
#                                     n_bins=20, names=state_lbls, avg_curve=False,
#                                     density=True, class1='Goal', plt_ttl=cal_ttl)
fig.savefig(froot + f'/../readme_imgs/goal_prediction_calibration.png')
