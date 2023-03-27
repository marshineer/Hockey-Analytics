import os
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
import torch.nn as nn
from torch import optim, Tensor, sigmoid, manual_seed, no_grad
from torch.utils.data import DataLoader
from models.common_torch import RegressionNN, CustomDataset, train_loop
from models.common_plot import plot_calibration_curves
from models.common import split_train_test_sets, create_stratify_feat


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the XGB hyperparameters and game state shot splits
n_itr = 500
ppath = f'/../data/goal_prob_model/hyperparameter_search_{n_itr}itr.pkl'
# ppath = f'/../data/goal_prob_model/hyperparameter_search_{n_itr}itr_coarse.pkl'
with open(froot + ppath, 'rb') as f:
    hyperparameter_data = pickle.load(f)
game_states = hyperparameter_data['game_states']
state_lbls = hyperparameter_data['state_lbls']
# strength_lbls = hyperparameter_data['strength_lbls']
# state_lbls = ['Even', 'PP', 'PK']
# n_states = len(state_lbls)
best_params = hyperparameter_data['cv_params']
n_best = 10

# Load the input data
info_path = f'/../data/goal_prob_model/expected_goal_modelling_info.pkl'
with open(froot + info_path, 'rb') as f:
    state_strength_info = pickle.load(f)
continuous_cols = state_strength_info['continuous_features']
boolean_features = state_strength_info['boolean_features']
new_booleans = state_strength_info['added_booleans']
boolean_cols = boolean_features + new_booleans
strength_features = state_strength_info['strength_lbls']
# # https://stackoverflow.com/questions/13016129/python-flatten-a-dict-of-lists-into-unique-values
# game_strength_list = [x for v in strength_feats.values() for x in v]
# all_dumb_losses = state_strength_info['strength_dumb_loss']
# game_strength_dumb_loss = [loss for loss in all_dumb_losses.values()]

# # Define the neural network template
# learning_rates = [list(np.arange(5e-4, 11e-4, 1e-4)),
#                   list(np.arange(5e-6, 16e-6, 1e-6)),
#                   list(np.arange(5e-5, 11e-5, 1e-5))]
# batch_sizes = [[16, 32, 64, 96, 128],
#                [8, 16, 24, 32, 48, 64],
#                [64, 96, 128, 192, 256]]
# # n_epochs_list = [15, 30, 5]
# n_epochs_list = [150, 10000, 500]
# loss_fn = nn.BCEWithLogitsLoss()

# Define the ensemble seeds
ens_seeds = [99669, 41975, 77840, 92597, 30427, 86846, 54781, 72793, 46298,
             26165, 99995, 10038, 37807, 52924, 99469, 49268, 40677, 41554,
             74175, 52253, 38737, 29474, 65014, 40201, 81510]
# ens_seeds = [99669, 41975, 77840, 92597, 30427,
#              86846, 54781, 72793, 46298, 26165]
# ens_seeds = [99669, 41975]
game_state_models = {state: [] for state in state_lbls}
# game_state_dumb_loss = []
for i, (state_shots, state_lbl) in enumerate(zip(game_states, state_lbls)):
    # # Omit irrelevant states
    # if state_lbl not in ['Even', 'PP', 'PK']:
    #     # state_lbls.remove(state_lbl)
    #     continue
    print(f'\nState: {state_lbl}')

    # Covert the shot list to a dataframe
    state_df = pd.DataFrame(state_shots, columns=list(state_shots[0].keys()))
    strength_cols = strength_features[state_lbl]
    # input_cols = cont_cols + bool_cols + strength_cols
    # input_df = state_df[input_cols]
    # target_df = state_df['goal']

    # # Calculate the dumb_loss
    # n_goals = len(state_df.loc[state_df.goal == True])
    # p_goal = n_goals / len(state_df)
    # dumb_loss = -(p_goal * np.log(p_goal) + (1 - p_goal) * np.log(1 - p_goal))
    # game_state_dumb_loss.append(dumb_loss)

    # # Define a column to stratify the data by game state and target value
    # strat_col = strength_cols + ['goal']
    # state_df['stratify'] = state_df.apply(lambda x:
    #                                       create_stratify_feat(x[strat_col]),
    #                                       axis=1)
    # # stratify_feat = state_df.stratify
    # stratify_feat = state_df.stratify.values

    for j, seed in enumerate(ens_seeds):
        # Set the numpy seed
        np.random.seed(seed)
        # print(f'\nSeed #{j + 1}: {seed}')

        # Split the data into stratified training and test sets
        tmp_split = split_train_test_sets(state_df, continuous_cols,
                                          boolean_cols + strength_cols,
                                          target='goal',
                                          stratify_feat='stratify',
                                          test_frac=0.15,
                                          shuffle_seed=seed,
                                          return_scaler=True)
        # # X_pd, y_pd = input_df.values, target_df.values
        # # tmp_split = train_test_split(X_pd, y_pd, test_size=0.25,
        # #                              random_state=seed, stratify=stratify_feat)
        # tmp_split = train_test_split(input_df, target_df, test_size=0.25,
        #                              random_state=seed, stratify=stratify_feat)
        X_train, X_test, y_train, y_test, x_scaler = tmp_split
        strength_test_inds = {}
        for col in strength_cols:
            strength_test_inds[col] = X_test.loc[X_test[col] == 1].index.tolist()

        t_start = time()
        hyper_param_models = []
        for params in best_params[state_lbl][:n_best]:
            # Initialize the XGBoost model
            # input_cols = continuous_cols + boolean_cols + strength_cols
            d_model = {'name': 'XGBoost Classifier',
                       'model': XGBClassifier(use_label_encoder=False,
                                              tree_method='hist',
                                              random_state=seed, **params),
                       'seed': seed,
                       'x_scaler': x_scaler,
                       'y_train': y_train.values,
                       'y_test': y_test.values,
                       # 'input_features': input_cols,
                       'strength_test_inds': strength_test_inds}

            # Train the model
            # model['model'].set_params(**params)
            d_model['model'].fit(X_train, y_train)
            d_model['train_pred'] = d_model['model'].predict_proba(X_train)[:, 1]
            d_model['train_loss'] = log_loss(y_train, d_model['train_pred'])

            # Test the model
            d_model['test_pred'] = d_model['model'].predict_proba(X_test)[:, 1]
            d_model['test_loss'] = log_loss(y_test, d_model['test_pred'])
            pred_cls = np.round(d_model['test_pred'], 0)
            test_acc = 100 * (pred_cls == y_test).sum() / y_test.size
            d_model['test_acc'] = test_acc
            hyper_param_models.append(d_model)
            # seed_models.append(model)

        t_train = time() - t_start
        print(f'Took {timedelta(seconds=t_train)} to train the {state_lbl} '
              f'XGBoost models (seed = {seed})')
        game_state_models[state_lbl] += hyper_param_models

# Save the models
fpath = f'/../data/goal_prob_model/goal_prob_game_state_ensembles_{n_itr}itr.pkl'
with open(froot + fpath, 'wb') as f:
    pickle.dump(game_state_models, f)
