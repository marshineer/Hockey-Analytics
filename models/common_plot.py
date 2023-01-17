import numpy as np
import matplotlib.pyplot as plt
from nhl_api.ref_common import game_time_to_sec
from torch import Tensor, sigmoid


def plot_calibration_curves(models, preds, ys, names=None, figsize=(8, 5),
                            n_bins=10, avg_curve=True, class1='Class 1',
                            plt_title=None):
    """ Plots calibration curves for a probabilistic binary classifier

    Reference: https://scikit-learn.org/stable/modules/calibration.html

    The function takes a list of models_and_analysis and their predictions and plots the
    calibration curve. This curve indicates whether the model is predicting class
    probabilities with a frequency proportional to the true fraction of that
    class in the probability bin. For example, it should predict class 1 with a
    probability of 80% correctly 80% of the time. Note: the calibration may be
    affected by sample size, for probability bins with few predictions.

    The prediction count distribution is plotted on a second axes, to allow
    visual identification of whether poor calibration corresponds to bins with
    few samples.

    Parameters
        models_and_analysis: list = models_and_analysis for which to plot calibration curve
        preds: list = test data predictions for each model
        ys: list = test data ground truths
        names: list = the model names, for plotting the legend
        figsize: tuple = the (width, height) size of the matplotlib figure
        n_bins: int = the number of probability bins used for calibration
        avg_curve: bool = indicates whether to average across models_and_analysis
        class1: str = the name of class 1, for plotting
        plt_title: str = the title of the plot (default to None)

    Returns
        fig: matplotlib figure
        ax: matplotlib axes = the true class 1 fraction (i.e. calibration)
        ax2: matplotlib axes = the prediction distribution
    """

    # Set generic names, if none provided
    if names is None:
        names = [f'Model {i + 1}' for i in range(len(models))]

    # Initialize the plot and create the names (if none given)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax.twinx()

    all_cnts = np.zeros((len(models), n_bins))
    all_mean_pred = np.zeros_like(all_cnts)
    all_true_frac = np.zeros_like(all_cnts)
    bins = np.linspace(0, 1, n_bins + 1)
    for i, (model, y_pred, y_true) in enumerate(zip(models, preds, ys)):
        # Calculate the calibration curve data
        # counts, bins = np.histogram(y_pred, bins=n_bins)
        # print(bins)
        counts, _ = np.histogram(y_pred, bins=bins)
        mean_pred = np.zeros(bins.size - 1)
        true_frac = np.zeros(bins.size - 1)
        for j in range(bins.size - 1):
            bin_mask = (y_pred >= bins[j]) & (y_pred < bins[j + 1])
            y_pred_bin = y_pred[bin_mask]
            mean_pred[j] = np.mean(y_pred_bin)
            y_true_bin = y_true[bin_mask]
            true_frac[j] = (y_true_bin == 1).sum() / y_true_bin.size
        all_cnts[i, :] = counts
        all_mean_pred[i, :] = mean_pred
        all_true_frac[i, :] = true_frac

    # Plot the calibration curve(s) and prediction distribution(s)
    xs = bins[:-1] + np.diff(bins)[0] / 2
    if avg_curve:
        ax.plot(np.mean(all_mean_pred, axis=0), np.mean(all_true_frac, axis=0),
                marker='s', label='Model Average')
        ax2.bar(xs, np.mean(all_cnts, axis=0), color='C1', width=1 / bins.size,
                alpha=0.4)
    else:
        for i, (y_pred, name) in enumerate(zip(preds, names)):
            ax.plot(all_mean_pred[i, :], all_true_frac[i, :], c=f'C{i}',
                    marker='s', label=name)
            # xs = bins[:-1] + np.diff(bins)[0] / 2
            if len(models) == 1:
                ax2.bar(xs, counts, color='C1', width=1 / bins.size, alpha=0.4)
            else:
                ax2.hist(y_pred, bins=bins, color=f'C{i}', histtype='step')

    # Plot the ideal calibration curve
    ax.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration')

    # Set plot attributes
    ax.set_xlabel(f"Mean Predicted Probability of '{class1}' in Bin",
                  fontsize=12)
    ax.set_ylabel(f"Actual Fraction of '{class1}' in Bin", fontsize=12)
    if plt_title is not None:
        ax.set_title(plt_title, fontsize=16)
    if len(models) <= 5 or avg_curve:
        ax.legend()
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.02))
    ax2.set_ylabel(f"Count of '{class1}' in Bin", fontsize=12)

    return fig, ax, ax2


def plot_in_game_probs(shots_df, home_win, team_names, x_scalers, models,
                       game_len=3600, names=None, avg_prob=True, figsize=(8, 5)):
    """ Plots in-game win probability predictions as function of game time

    The probability of a home win is predicted using the game state at each
    second of a game. The game state include the model inputs: goal differential,
    shot differential and game time.

    Ideally, late-game events have a stronger effect on predicted probability
    than early-game events, the game time will have a drift effect, and goals
    will cause jumps in the probability. The probability in the last second of
    the game should always be 1 or 0.

    Parameters
        shots_df: dataframe = all shots for the game
        home_win: bool = indicates whether home team wins
        team_names: list = team names ['home team', 'away team']
        x_scalers: list = scalers fit to the training data
        models_and_analysis: list = models_and_analysis for which to plot calibration curve
        game_len: int = length of the game (seconds)
        names: list = the model names, for plotting the legend
        avg_prob: bool = indicates whether to average the model predictions
        figsize: tuple = the (width, height) size of the matplotlib figure
    """

    # Set generic names, if none provided
    if names is None:
        names = [f'Model {i + 1}' for i in range(len(models))]

    # Generate a list of inputs
    shot_list = shots_df.to_dict('records')
    input_arr = np.zeros((game_len, 3))
    home_shots = 0
    away_shots = 0
    home_score = 0
    away_score = 0
    shot_cnt = 0
    shot = shot_list[shot_cnt]
    period = shot['period']
    period_time = shot['period_time']
    shot_time = (period - 1) * 20 * 60 + game_time_to_sec(period_time)
    goal_list = []
    for sec in range(game_len):
        # Set the game state (model input)
        input_list = [home_score - away_score, home_shots - away_shots, sec]
        input_arr[sec, :] = input_list

        # If a shot occurred at this time, update the game state
        if sec == shot_time:
            # Set the game state values
            if shot['shooter_home']:
                home_shots += 1
            else:
                away_shots += 1
            home_score = shot['home_score']
            away_score = shot['away_score']

            # If the shot was a goal, add to plot list
            if shot['shot_result'] == 'GOAL':
                score1 = home_score if home_win else away_score
                score2 = away_score if home_win else home_score
                goal_list.append([shot_time, shot['shooter_home'],
                                  f'{score1}-{score2}'])

            # Get the next shot
            shot_cnt += 1
            if shot_cnt < len(shot_list):
                shot = shot_list[shot_cnt]
            period = shot['period']
            period_time = shot['period_time']
            shot_time = (period - 1) * 20 * 60 + game_time_to_sec(period_time)

    # Scale the inputs and make predictions
    in_game_prob = np.zeros((game_len, len(models)))
    for i, model in enumerate(models):
        x_scaler = x_scalers[i]
        input_arr_scaled = x_scaler.transform(input_arr)
        model_output = sigmoid(model['model'](Tensor(input_arr_scaled)))
        in_game_prob[:, i] = model_output.detach().numpy().squeeze()

    # Plot the in-game 'home win' probabilities
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if avg_prob:
        ax.plot(np.arange(game_len), np.mean(in_game_prob, axis=-1))
    else:
        for i in range(len(models)):
            ax.plot(np.arange(game_len), in_game_prob[:, i], f'C{i}',
                    label=names[i])
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.03))

    # Plot the goal times
    prev_t = -220
    prev_home = None
    for t_goal, home_goal, score in goal_list:
        ax.vlines(t_goal, 0, 1, 'k', '--', linewidth=0.5)
        if home_goal:
            height = 0.95
            if (t_goal - prev_t) < 220 and prev_home == home_goal:
                height += 0.035
        else:
            height = 0.05
            if (t_goal - prev_t) < 220 and prev_home == home_goal:
                height -= 0.035
        ax.text(t_goal, height, score, fontsize=10, ha='left', va='center')
        prev_t = t_goal
        prev_home = home_goal

    # Set plot attributes
    ax.set_xlabel('Game Time (seconds)', fontsize=12)
    ax.set_ylabel(f'Probability of {team_names[0]} Win', fontsize=12)
    ax.set_ylim([0, 1])
    w_team = team_names[0] if home_win else team_names[1]
    score1 = home_score if home_win else away_score
    score2 = away_score if home_win else home_score
    ttl_str1 = f'{team_names[1]} at {team_names[0]}\n'
    ttl_str2 = f'Final score {score1}-{score2} {w_team}'
    ax.set_title(ttl_str1 + ttl_str2, fontsize=16)

    return fig, ax
