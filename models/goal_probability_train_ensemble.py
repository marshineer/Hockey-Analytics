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


# Dynamically set the CWD
froot = str(os.path.dirname(__file__))

# Load the XGB hyperparameters and game state data splits
n_itr = 1000
pre_model_path = f'/../data/goal_prob_model/pre_model_data_{n_itr}itr.pkl'
with open(froot + pre_model_path, 'rb') as f:
    pre_model_data = pickle.load(f)
game_states = pre_model_data['game_states']
state_lbls = pre_model_data['state_lbls']
xgb_params = pre_model_data['xgb_params']
feat_cols = pre_model_data['features']

# Define the neural network template
learning_rates = [list(np.arange(5e-4, 11e-4, 1e-4)),
                  list(np.arange(5e-6, 16e-6, 1e-6)),
                  list(np.arange(5e-5, 11e-5, 1e-5))]
batch_sizes = [[16, 32, 64, 96, 128],
               [8, 16, 24, 32, 48, 64],
               [64, 96, 128, 192, 256]]
# n_epochs_list = [15, 30, 5]
n_epochs_list = [150, 10000, 500]
loss_fn = nn.BCEWithLogitsLoss()

# Define the ensemble seeds
# ens_seeds = [99669, 41975, 77840, 92597, 30427, 86846, 54781, 72793, 46298,
#              26165, 99995, 10038, 37807, 52924, 99469, 49268, 40677, 41554,
#              74175, 52253, 38737, 29474, 65014, 40201, 81510]
ens_seeds = [99669, 41975, 77840, 92597, 30427,
             86846, 54781, 72793, 46298, 26165]
# ens_seeds = [99669, 41975]
game_state_models = []
game_state_dumb_loss = []
for i, seed in enumerate(ens_seeds):
    # Set the pytorch and numpy seeds
    # https://pytorch.org/docs/stable/notes/randomness.html
    manual_seed(seed)
    np.random.seed(seed)
    print(f'\nSeed #{i + 1}: {seed}')

    nn_cnt = 0
    state_ys = []
    for j, (state_ls, lbl) in enumerate(zip(game_states, state_lbls)):
        # Covert the shot list to a dataframe
        state_df = pd.DataFrame(state_ls)
        state_df = state_df[feat_cols]

        # Calculate the dumb logloss for the game state
        p_goal = len(state_df.loc[state_df.goal == True]) / len(state_df)
        dum_loss = -(p_goal * np.log(p_goal) + (1 - p_goal) * np.log(1 - p_goal))
        if i == 0:
            game_state_dumb_loss.append(dum_loss)

        # Split the data into stratified training and test sets
        X_pd, y_pd = state_df.iloc[:, :-1].values, state_df.iloc[:, -1].values
        tmp_split = train_test_split(X_pd, y_pd, test_size=0.25,
                                     random_state=seed, stratify=y_pd)
        X_train, X_test, y_train, y_test = tmp_split

        # Scale the numerical data (first 8 feature columns are numerical)
        n_num_cols = 9
        x_scaler = MinMaxScaler()
        X_train_norm = x_scaler.fit_transform(X_train[:, :n_num_cols])
        X_train[:, :n_num_cols] = X_train_norm
        X_test_norm = x_scaler.transform(X_test[:, :n_num_cols])
        X_test[:, :n_num_cols] = X_test_norm
        train_data = CustomDataset(X_train, y_train)
        test_data = CustomDataset(X_test, y_train)

        t_start = time()
        # XGBoost model
        if j < 3:
            hyper_param_models = []
            for params in xgb_params[j]:
                # Initialize the XGBoost model
                model = {'name': 'XGBoost Classifier',
                         'model': XGBClassifier(use_label_encoder=False,
                                                tree_method='hist', **params),
                         'seed': seed,
                         'x_scaler': x_scaler,
                         'y_train': y_train,
                         'y_test': y_test}

                # Train the model
                # model['model'].set_params(**params)
                model['model'].fit(X_train, y_train)
                model['train_pred'] = model['model'].predict_proba(X_train)[:, 1]
                model['train_loss'] = log_loss(y_train, model['train_pred'])

                # Test the model
                model['test_pred'] = model['model'].predict_proba(X_test)[:, 1]
                model['test_loss'] = log_loss(y_test, model['test_pred'])
                pred_cls = np.round(model['test_pred'], 0)
                test_acc = 100 * (pred_cls == y_test).sum() / y_test.size
                model['test_acc'] = test_acc
                hyper_param_models.append(model)
                # seed_models.append(model)

            t_train = time() - t_start
            print(f'Took {timedelta(seconds=t_train)} to train the {lbl} '
                  f'XGBoost models')
            # seed_models += hyper_param_models
            if i == 0:
                game_state_models.append(hyper_param_models)
            else:
                game_state_models[j] += hyper_param_models
        # Neural network model
        else:
            # Initialize the neural network model
            model = RegressionNN(X_train.shape[-1], 16, 1).to('cpu').float()

            # Define the hyperparameters
            lr = random.choice(learning_rates[nn_cnt])
            train_bs = random.choice(batch_sizes[nn_cnt])
            n_epochs = n_epochs_list[nn_cnt]
            params = [lr, train_bs, n_epochs]
            nn_cnt += 1

            # Create the data loaders
            train_dataloader = DataLoader(train_data, batch_size=train_bs)
            test_dataloader = DataLoader(test_data, batch_size=X_test.shape[-1])
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train the model
            # train_loss_avg = np.zeros(n_epochs)
            # train_loss_std = np.zeros(n_epochs)
            for t in range(n_epochs):
                epoch_loss = train_loop(train_dataloader, model, loss_fn,
                                        optimizer)
            #     train_loss_avg[t] = np.mean(epoch_loss)
            #     train_loss_std[t] = np.std(epoch_loss)
            # # Plot the progression of loss and accuracy
            # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            # ax.plot(np.arange(n_epochs), train_loss_avg)
            # ax.fill_between(np.arange(n_epochs),
            #                 train_loss_avg - train_loss_std,
            #                 train_loss_avg + train_loss_std,
            #                 alpha=0.3)
            # ax.set_xlabel('Epoch', fontsize=12)
            # ax.set_ylabel('Training Loss', fontsize=12)
            # fig.tight_layout()
            # plt.show()

            # Test the model
            # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            # with no_grad():
            #     for X, y in test_dataloader:
            #         test_pred = model(X)
            #         test_loss = loss_fn(test_pred, y).item()
            train_pred = sigmoid(model(Tensor(X_train.astype(float)))).detach()
            train_pred = train_pred.squeeze().numpy()
            train_loss = log_loss(y_train, train_pred, eps=1e-7)
            test_pred = sigmoid(model(Tensor(X_test.astype(float)))).detach()
            test_pred = test_pred.squeeze().numpy()
            test_loss = log_loss(y_test, test_pred, eps=1e-7)
            pred_cls = np.round(test_pred, 0)
            test_acc = 100 * (pred_cls == y_test).sum() / y_test.size
            model_dict = {'name': 'Neural Network',
                          'model': model,
                          'seed': seed,
                          'params': params,
                          'x_scaler': x_scaler,
                          'y_train': y_train,
                          'y_test': y_test,
                          'train_pred': train_pred,
                          'train_loss': train_loss,
                          'test_pred': test_pred,
                          'test_loss': test_loss,
                          'test_acc': test_acc}
            # seed_models.append(model_dict)
            if i == 0:
                game_state_models.append([model_dict])
            else:
                game_state_models[j].append(model_dict)

            t_train = time() - t_start
            print(f'Took {timedelta(seconds=t_train)} to train the {lbl} '
                  f'Neural Network model')

# Save the models
fpath = f'/../data/goal_prob_model/goal_prob_game_state_ensembles_{n_itr}itr.pkl'
with open(froot + fpath, 'wb') as f:
    pickle.dump(game_state_models, f)

# # Plot the prediction distributions
# fig, ax = plt.subplots(1, 1, figsize=(6, 4))
# for model, lbl in zip(game_state_models, state_lbls):
#     ax.hist(model['test_pred'], bins=50, density=True, histtype='step',
#             label=lbl)
# ax.set_title('XGBoost Prediction Distribution', fontsize=16)
# ax.set_xlabel("Probability of 'Goal'", fontsize=12)
# ax.set_ylabel('Density', fontsize=12)
# fig.savefig(froot + f'/../readme_imgs/goal_prediction_distributions.png')

# Calculate the values for plotting
game_state_losses = []
game_state_preds = []
game_state_ys = []
for d_state in game_state_models:
    game_state_losses.append([model['test_loss'] for model in d_state])
    state_preds = d_state[0]['test_pred']
    state_ys = d_state[0]['y_test']
    for model in d_state[1:]:
        state_preds = np.hstack((state_preds, model['test_pred']))
        state_ys = np.hstack((state_ys, model['y_test']))
    game_state_preds.append(state_preds)
    game_state_ys.append(state_ys)

# Plot the training loss and trivial loss for each state
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
xs = np.arange(len(game_states)) + 0.5
ax.scatter(xs, game_state_dumb_loss, c='k', marker='*', label='Dumb Loss')
ax.violinplot(game_state_losses, xs, showmeans=True)
ax.set_xlabel('Game State', fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels(state_lbls, fontsize=10)
ax.set_ylabel('Logloss', fontsize=12)
ax.legend(loc=2)
fig.savefig(froot + f'/../readme_imgs/goal_prediction_loss_violin.png')

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(xs, game_state_dumb_loss, c='k', marker='*', label='Dumb Loss')
ax.boxplot(game_state_losses, positions=xs, showmeans=True)
ax.set_xlabel('Game State', fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels(state_lbls, fontsize=10)
ax.set_ylabel('Logloss', fontsize=12)
ax.legend(loc=2)
fig.savefig(froot + f'/../readme_imgs/goal_prediction_loss_box.png')

# Plot the calibration curves
cal_ttl = 'Expected Goal Calibration Curves'
fig, _, _ = plot_calibration_curves(game_state_preds, game_state_ys,
                                    names=state_lbls, avg_curve=False,
                                    class1='Goal', plt_ttl=cal_ttl)
fig.savefig(froot + f'/../readme_imgs/goal_prediction_calibration.png')
