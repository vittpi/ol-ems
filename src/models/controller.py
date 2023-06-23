from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import argparse
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpylayers.torch import CvxpyLayer

from src.dataloader.csv_dataset import CSVDataset
from src.utils import denormalise

import logging
logging = logging.getLogger('pytorch_lightning')


VARIABLES = ['Ps', 'Pl', 'Pr', 'Pg', 'eps']


class OptimisationLayer(nn.Module):
    """This class is used to define the optimisation problem.

    Args:
        prediction_horizon (int): Prediction horizon.
        A_matrix (torch.Tensor): A matrix of the storage dynamics.
        B_matrix (torch.Tensor): B matrix of the storage dynamics.
        charge_min_max (Tuple[float, float]): Minimum and maximum storage charge.
        storage_power_min_max (Tuple[float, float]): Minimum and maximum storage power.
        grid_power_min_max (Tuple[float, float]): Minimum and maximum grid power.

    The optimisation problem is defined as a neural network that uses the parameters
    predicted by the controller to solve the optimisation problem.
    """

    def __init__(self, prediction_horizon: int,
                 A_matrix: torch.Tensor,
                 B_matrix: torch.Tensor,
                 charge_min_max: Tuple[float, float],
                 storage_power_min_max: Tuple[float, float],
                 grid_power_min_max: Tuple[float, float]) -> None:
        super(OptimisationLayer, self).__init__()
        # Check that min < max
        assert charge_min_max[0] < charge_min_max[
            1], f"The minimum charge must be smaller than the maximum charge. Got {charge_min_max[0]} and {charge_min_max[1]}."
        assert storage_power_min_max[0] < storage_power_min_max[
            1], f"The minimum storage power must be smaller than the maximum storage power. Got {storage_power_min_max[0]} and {storage_power_min_max[1]}."
        assert grid_power_min_max[0] < grid_power_min_max[
            1], f"The minimum grid power must be smaller than the maximum grid power. Got {grid_power_min_max[0]} and {grid_power_min_max[1]}."

        self._prediction_horizon = prediction_horizon

        # Define the variables
        s = Variable((1, prediction_horizon+1))  # Storage charge
        Ps_in = Variable((1, prediction_horizon))  # Storage power in
        Ps_out = Variable((1, prediction_horizon))  # Storage power out
        Pl = Variable((1, prediction_horizon))  # Load power
        Pr = Variable((1, prediction_horizon))  # Renewable generator power
        eps = Variable((1, prediction_horizon))  # Soft constraint

        # Define the parameters
        price = Parameter((1, prediction_horizon))
        load = Parameter((1, prediction_horizon))
        pv = Parameter((1, prediction_horizon))
        s0 = Parameter((1, 1))

        # Define the objective function and constraints
        objective = 0.0
        constraints = []
        for k in range(prediction_horizon):
            objective += -1 * price[0, k] * \
                (Pr[:, k]-Pl[:, k] - (Ps_in[:, k] - Ps_out[:, k])) + 10000*(eps[:, k])
            constraints += [
                s[:, k+1] == A_matrix @ s[:, k] +
                B_matrix[0,0] * Ps_in[:, k] + B_matrix[0,1] * Ps_out[:, k],  # Storage dynamics
                s[:, k] >= charge_min_max[0],  # Storage min charge limit
                s[:, k] <= charge_min_max[1],  # Storage max charge limit
                # Storage min power limit
                Ps_in[:, k] <= storage_power_min_max[1],
                Ps_in[:, k] >= 0,
                Ps_out[:, k] <= storage_power_min_max[1],
                Ps_out[:, k] >= 0, 
                Pl[:, k] >= load[0, k] - eps[:, k],  # Load lower bounds
                Pl[:, k] <= load[0, k] + eps[:, k],  # Load upper bounds
                Pr[:, k] >= pv[0, k] - eps[:, k],  # Renewable lower bounds
                Pr[:, k] <= pv[0, k] + eps[:, k],  # Renewable upper bounds
                Pr[:, k] - Pl[:, k] - \
                Ps_in[:, k] + Ps_out[:, k] >= grid_power_min_max[0],  # Max grid power
                Pr[:, k] - Pl[:, k] - \
                Ps_in[:, k] + Ps_out[:, k] <= grid_power_min_max[1]  # Min grid power
            ]
        constraints += [s[:, 0] == s0]  # Feedback constraint

        # Define the problem
        problem = Problem(Minimize(objective), constraints)
        self._optimisation_layer = CvxpyLayer(problem,
                                              parameters=[price, load, pv, s0],
                                              variables=[s, Ps_in, Ps_out, Pl, Pr, eps])

    def forward(self, price_hat: torch.Tensor,
                load_hat: torch.Tensor,
                pv_hat: torch.Tensor,
                initial_state: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the optimisation problem.

        Args:
            price_hat (torch.Tensor): Predicted price.
            load_hat (torch.Tensor): Predicted load.
            pv_hat (torch.Tensor): Predicted pv.
            initial_state (torch.Tensor): Initial state of the storage.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The solution of the optimisation problem.
        """
        batch_size = price_hat.shape[0]
        device = price_hat.device
        assert price_hat.shape[0] == load_hat.shape[
            0], f"The batch size of the price and load predictions must be the same. Got {price_hat.shape[0]} and {load_hat.shape[0]}."
        assert price_hat.shape[0] == pv_hat.shape[
            0], f"The batch size of the price and pv predictions must be the same. Got {price_hat.shape[0]} and {pv_hat.shape[0]}."

        s = torch.zeros(
            (batch_size, self._prediction_horizon+1), device=device, dtype=torch.float64)
        Ps_in = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        Ps_out = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        Pl = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        Pr = torch.zeros((batch_size, self._prediction_horizon),
                         device=device, dtype=torch.float64)
        eps = torch.zeros(
            (batch_size, self._prediction_horizon), device=device, dtype=torch.float64)

        # Compute solution for each input in the batch
        for i in range(batch_size):
            s[i, :], Ps_in[i, :], Ps_out[i, :], Pl[i, :], Pr[i, :], eps[i, :] = \
                self._optimisation_layer(price_hat[i:i+1, :], load_hat[i:i+1, :],
                                         pv_hat[i:i+1, :], initial_state[i:i+1, 0:1])

        return Ps_in, Ps_out, Pl, Pr, eps


class Controller(nn.Module):
    """This class is used to define the controller for the system.

    The controller is a neural network that estimates the parameters of the optimisation problem
    and uses the predicted parameters to solve the optimisation problem.

    Args:
        optimisation_layer (OptimisationLayer): The optimisation layer.
        dataset (CSVDataset): The dataset to be used.
        hidden_dim (int): The dimension of the hidden layers.
        num_layers (int): The number of layers of the neural network.
    """

    def __init__(self,
                 optimisation_layer: OptimisationLayer,
                 dataset: CSVDataset,
                 hidden_dim: int,
                 num_layers: int,
                 ) -> None:
        super(Controller, self).__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

        self._lookback_window = dataset.lookback_window
        self._prediction_horizon = dataset.prediction_horizon
        self._scaling_factors_min = dataset.scaling_factors_min
        self._scaling_factors_max = dataset.scaling_factors_max
        self._output_features = dataset.output_features
        self._available_features = dataset.available_features
        self._bypass_features = dataset.bypass_features
        self._indices_bypass_features = dataset.indices_bypass_features
        self._input_dim = len(dataset.input_features)
        self._output_dim = len(dataset.output_features)

        self._lstm = nn.LSTM(self._input_dim, self._hidden_dim,
                             self._num_layers, batch_first=True, bias=True)

        self._fc = nn.Linear(
            self._hidden_dim, self._output_dim*self._prediction_horizon)

        self._optimisation_layer = optimisation_layer

    def _select_parameters_and_denormalise(self, 
                                           predictions: torch.Tensor,
                                           bypass: torch.Tensor
                                           ) -> torch.Tensor:
        """This function is used to route the correct variables into the optimisation layer.
            The bypass features are assumed to be known by the optimisation layer.
            The features listed in the output features are the `predictions` of the
            neural network. The other features are in the `bypass` vector.
            Data are denormalised before being routed to the optimisation layer.

        Args:
            predictions (torch.Tensor): The predictions of the neural network.
            bypass (torch.Tensor): The data that are not predicted by the neural network
                                    which are assumed to be known by the optimisation.
        """
        batch_size = predictions.shape[0]
        all_features = torch.zeros(batch_size,
                                   self._prediction_horizon,
                                   len(self._available_features),
                                   device=predictions.device,
                                   dtype=torch.float64)

        for i, feature in enumerate(self._available_features):
            if feature in self._output_features:
                # Feature is in the predicted features
                index = self._output_features.index(feature)
                all_features[:, :, i] = predictions[:, :, index]
            else:
                # Feature is in the bypass features
                index = self._bypass_features.index(feature)
                all_features[:, :, i] = bypass[:, :, index]

        # Denormalise data
        assert len(self._scaling_factors_min) == len(self._available_features)
        all_features = denormalise(
            all_features, self._scaling_factors_min, self._scaling_factors_max)

        # Select data with respect to their meaning
        pv_hat = all_features[:, :, 0]
        load_hat = all_features[:, :, 1]
        price_hat = all_features[:, :, 2]
        return price_hat, load_hat, pv_hat

    def forward(self,
                inputs: torch.Tensor,
                bypass: torch.Tensor,
                initial_state: torch.Tensor
                ) -> Tuple[torch.Tensor,
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the controller.

        It returns the predictions for the features with respect to the `prediction_horizon`. These
        are the inputs of the optimisation problem. The output then also contains the optimal solution
        of the optimisation problem.
        """
        predictions, _ = self._lstm(inputs)
        predictions = self._fc(predictions[:, -1, :])
        predictions = predictions.reshape(predictions.shape[0],
                                          self._prediction_horizon,
                                          self._output_dim)

        # Select the parameters from the predictions and the bypass, denormalise
        # them and send them to the optimisation layer
        price_hat, load_hat, pv_hat = self._select_parameters_and_denormalise(
            predictions, bypass)

        Ps_in, Ps_out, Pl, Pr, eps = self._optimisation_layer(
            price_hat, load_hat, pv_hat, initial_state)

        return predictions, (Ps_in, Ps_out, Pl, Pr, eps)

    @ staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This method is used to add the model specific arguments to the parent parser."""
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=2,
                            help="The dimension of the hidden layers.")
        parser.add_argument("--num_layers", type=int, default=1,
                            help="The number of layers.")
        return parser
