
import logging
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import torch
from pathlib import Path
from src.dataloader.csv_dataset import CSVDataset
from src.training_and_evaluation.learner import METRICS_TENDENCY
from src.utils import denormalise
import pandas as pd
import os
import tikzplotlib
plt.style.use('ieee')
logging = logging.getLogger('pytorch_lightning')


def plot_predictions(experiment_path: str,
                     dataset: CSVDataset,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     output_features: List[str],
                     t_initial: int, t_final: int,
                     save_folder_name: str = "predictions",
                     data_split: str = "test") -> None:
    """This method is used to plot the predictions vs. ground truth.

    Args:
        dataset (CSVDataset): the dataset to obtain the grounf truth profiles.
        predictions (torch.Tensor): the predicted profiles.
        targets (torch.Tensor): the ground truth profiles.
        output_features (List[str]): the features that have been predicted.
        t_initial (int): the initial time step to be plotted.
        t_final (int): the final time step to be plotted.
        save_folder_name (str): the name of the folder where the plots will be saved.
        data_split (str): The data split to be used for the plot.
    """
    # Get ground truth profiles
    profiles = targets

    # Create predictions folder
    Path(os.path.join(experiment_path, save_folder_name, data_split)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name, data_split)

    prediction_horizon = predictions.shape[1]
    for prediction_sample in range(prediction_horizon):
        # Get predicted profiles and rescale them
        profiles_hat = predictions[:, prediction_sample, :]
        profiles_targets = profiles[:, prediction_sample, :]

        scaling_factors_min = dataset.scaling_factors_min[dataset.indices_output_features]
        scaling_factors_max = dataset.scaling_factors_max[dataset.indices_output_features]

        profiles_hat = denormalise(
            profiles_hat, scaling_factors_min, scaling_factors_max)
        profiles_targets = denormalise(
            profiles_targets, scaling_factors_min, scaling_factors_max)

        # Plot the profiles
        fig, axs = plt.subplots(
            len(dataset.available_features), 1, figsize=(10, 6))
        fig.suptitle('Predictions vs. Ground Truth')
        for i in range(len(dataset.available_features)):
            # Subplot title
            axs[i].set_title(dataset.available_features[i])
            # Plot the ground truth profile
            axs[i].grid()
            # Plot predicted profiles
            if dataset.available_features[i] in output_features:
                # if the profile is in the predicted features, plot the prediction
                idx = output_features.index(dataset.available_features[i])
                axs[i].plot(profiles_targets[t_initial:t_final, idx], label="Ground Truth" if i == len(
                dataset.available_features) - 1 else None)
                axs[i].plot(profiles_hat[t_initial:t_final, idx], label='Prediction' if i == len(
                    dataset.available_features) - 1 else None)

        # Add the legend horizontally below the subplots, outside the figure
        axs[-1].legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=5)
        # Save figure
        if save_folder_name == "predictions":
            plt.savefig(os.path.join(
                save_directory,
                f"predictions_prediction_sample_{prediction_sample}.png"), bbox_inches='tight')
            plt.close(fig)
            plt.clf()
        if save_folder_name is not "predictions" and prediction_sample == 0:
            plt.savefig(os.path.join(save_directory,
                f"predictions_prediction_sample_{prediction_sample}.png"), bbox_inches='tight')
            tikzplotlib.save(os.path.join(save_directory,
                f"predictions_prediction_sample_{prediction_sample}.tex"))
            plt.close(fig)

def plot_decision_variables(experiment_path: str,
                            decision_variables: Dict[str, torch.Tensor],
                            constraints: Dict[str, List[Union[float, torch.Tensor]]],
                            data_split: str) -> None:
    """Plot the decision variables.

    Args:
        experiment_path (str): Path to the experiment folder.
        decision_variables (Dict[str, torch.Tensor]): Dictionary of decision variables.
        constraints (Dict[str, List[Union[float, torch.Tensor]]]): Dictionary of constraints.
        data_split (str): Data split.
    """
    prediction_horizon = decision_variables['Ps_in'].shape[1]

    # Create decisions folder
    Path(os.path.join(experiment_path, "decisions", data_split)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, "decisions", data_split)

    for prediction_sample in range(prediction_horizon):
        # Create figure
        fig, axs = plt.subplots(len(decision_variables), figsize=(15, 10))
        fig.suptitle('Decision Variables')
        # Plot decision variables
        for i, (key, value) in enumerate(decision_variables.items()):
            # Subplot title
            axs[i].set_title(key)
            # Plot the decision variable
            axs[i].plot(value[:, prediction_sample])
            # Plot constraints
            constr = constraints[key]
            for j in range(len(constr)):
                # Plot horizontal line for grid and storage
                if type(constr[j]) == float:
                    axs[i].axhline(y=constr[j], color='r',
                                   linestyle='dashed')
                # Plot profile for renewable and load
                elif type(constr[j]) == torch.Tensor and data_split == "test":
                    axs[i].plot(constr[j], color='r',
                                linestyle='dashed')
            axs[i].grid()
        # Save figure
        plt.savefig(os.path.join(save_directory,
                    f"decision_variables_prediction_sample_{prediction_sample}.png"), bbox_inches='tight')
        plt.close()
        plt.clf()


def plot_trajectory(experiment_path: str,
                    state: torch.Tensor, 
                    constraints_state: Tuple[float, float],
                    controller_outputs: torch.Tensor,
                    constraints_input: Tuple[float, float],
                    t_initial: int, t_final: int,
                    save_folder_name: str = "trajectory") -> None:
    """This function is used to plot the state and controller output trajectories.

    Args:
        experiment_path (str): Path to the experiment folder.
        state (torch.Tensor): State trajectory.
        constraints_state (List): List of state constraints.
        controller_outputs (torch.Tensor): Outputs trajectory.
        constraints_input (List): List of input constraints. 
        t_initial (int): Initial time step to plot.
        t_final (int): Final time step to plot.
        folder_name (str): Name of the folder where to save the plots.
    """
    # Configure plot
    constraints_color = 'r'
    constraints_style = 'dashed'

    # Create trajectory folder
    Path(os.path.join(experiment_path, save_folder_name)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name)

    # Create figure
    fig, axs = plt.subplots(
        state.shape[1] + controller_outputs.shape[1], figsize=(15, 10))
    # State plot
    for i in range(state.shape[1]):
        axs[i].set_title('State')
        axs[i].plot(state[t_initial:t_final, i].cpu().numpy())
        for j in range(len(constraints_state)):
            axs[i].axhline(y=constraints_state[j],
                        color=constraints_color,
                        linestyle=constraints_style)
        axs[i].grid()
    # Input plot
    for i in range(controller_outputs.shape[1]):
        j = i + state.shape[1]
        axs[j].set_title('Input')
        axs[j].plot(controller_outputs[t_initial:t_final-1, i])
        for k in range(len(constraints_input[i])):
            axs[j].axhline(y=constraints_input[i][k],
                           color=constraints_color,
                           linestyle=constraints_style)
        axs[j].grid()
    # Save figure
    plt.savefig(os.path.join(save_directory,
                "state_and_input.png"), bbox_inches='tight')
    if save_folder_name is not "trajectory":
        tikzplotlib.save(os.path.join(save_directory,"state_and_input.tex"))
    plt.close()


def total_cost(price: torch.Tensor, Pg: torch.Tensor) -> torch.Tensor:
    """This function computes the total cost of the power generation.
    Args:
        price (torch.Tensor): electricity price in [€/MWh].
        Pg (torch.Tensor): power generation in [kW].
    Returns:
        float: total cost in [€] (sample time assumed to be 1h).
    """
    return -1*torch.sum(price*Pg).item()/1000  # [€]


def plot_and_log_total_cost(experiment_path: str,
                            price: torch.Tensor,
                            Pg: torch.Tensor,
                            results: Dict[str, Any],
                            data_split: str,
                            prediction_horizon: int) -> None:
    """This function is used to plot the cost for all the prediction horizons.

    Args:
        experiment_path (str): Path to the experiment folder.
        price (torch.Tensor): Price.
        Pg (torch.Tensor): Power generation.
        results (Dict[str, Any]): Dictionary of results.
        data_split (str): Data split.
        prediction_horizon (int): Prediction horizon.
    """

    # Create cost folder
    Path(os.path.join(experiment_path, "cost", data_split)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, "cost", data_split)

    results['total_cost'] = {}
    total_costs = []
    logging.info("Total cost for data split %s:", data_split)
    for prediction_sample in range(prediction_horizon):
        try:
            results['total_cost'][prediction_sample] = total_cost(
                price, Pg[:, prediction_sample])
        except:
            results['total_cost'][prediction_sample] = 0
        total_costs.append(results['total_cost'][prediction_sample])
        logging.info(
            "Total cost for prediction sample %d: %f", prediction_sample, results['total_cost'][prediction_sample])

    # Create figure
    fig, axs = plt.subplots(1, figsize=(15, 10))
    fig.suptitle('Total Cost')
    x = np.arange(prediction_horizon)
    y = total_costs

    plt.plot(x, y)
    plt.xlabel('Prediction Sample')
    plt.ylabel('Total Cost [€/kWh]')
    plt.grid()
    plt.savefig(os.path.join(save_directory,
                f"total_cost.png"), bbox_inches='tight')
    plt.close()
    plt.clf()


def plot_different_runs_and_metrics(results_df: pd.DataFrame, experiment_path: str) -> None:
    """Plot each different run separately on the same graph with respect to the iteration and all the logged metrics."""
    # Get unique runs
    # Color each run differently
    unique_runs = results_df["trial_id"].unique()
    # Get all the metrics from the results dataframe which where the columns can be found in the METRICS_TENDENCY dictionary's keys
    unique_metrics = [
        column for column in results_df.columns if column in METRICS_TENDENCY]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_runs)))
    for metric in unique_metrics:
        last_value = None
        best_id = None
        ymin = float("inf")
        ymax = float("-inf")
        for i, run in enumerate(unique_runs):
            run_df = results_df[results_df["trial_id"] == run]
            x = np.arange(0, len(run_df[metric].values))
            plt.plot(x, run_df[metric].values, color=colors[i])
            ymin = min(ymin, np.min(run_df[metric].values))
            ymax = max(ymax, np.max(run_df[metric].values))
            if last_value is None:
                last_value = run_df[metric].values[-1]
                best_id = run
            else:
                if METRICS_TENDENCY[metric] == "min":
                    if run_df[metric].values[-1] < last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run
                else:
                    if run_df[metric].values[-1] > last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run

        # Plot the best run with respect to black color
        best_run = results_df[results_df["trial_id"] == best_id]
        x = np.arange(0, len(best_run[metric].values))
        plt.plot(x, best_run[metric].values, color="black")
        plt.xlabel("Tuning step")
        plt.ylabel(metric)
        plt.grid()
        plt.ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)
        plt.title(
            f"Individual runs, All runs: {len(unique_runs)}, Best run id: {best_id}", fontsize=10)
        plt.savefig(os.path.join(experiment_path, f"{metric}_individual_runs.png"),
                    bbox_inches="tight")
        plt.close()
        plt.clf()
