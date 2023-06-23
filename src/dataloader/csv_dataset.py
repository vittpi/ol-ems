from src.utils import normalise
from typing import Tuple, List, Any, Dict
from torch.utils.data import Dataset
import pandas as pd
import torch
import argparse
import logging
logging = logging.getLogger('pytorch_lightning')


class CSVDataset(Dataset):
    """This class is used to create a dataset from a .csv file.

    Args:
        site_id (int): ID of the microgrid site.
        lookback_window (int): Number of time steps to look back.
        prediction_horizon (int): Prediction horizon time steps into the future.
        train_batchsize (int): Number of training samples for each timestep.
        valid_batchsize (int): Number of validation or test samples for each timestep.
        input_features (List[str]): List of input features to use.
        output_features (List[str]): List of output features to predict.
    """

    def __init__(self,
                 site_id: int,
                 lookback_window: int,
                 prediction_horizon: int,
                 train_batchsize: int,
                 valid_batchsize: int,
                 input_features: List = [],
                 output_features: List = [],
                 ) -> None:

        self._data = pd.read_csv('data/site_' + str(site_id) + '/dataset.csv')
        # First column is the timestamp
        self.available_features = list(self._data.columns.values[1:])

        # Check if the input and output features are available in the `available_features` list
        for input_feature in input_features:
            if input_feature not in self.available_features:
                raise ValueError(
                    f"Input feature {input_feature} is not available.")

        for output_feature in output_features:
            if output_feature not in self.available_features:
                raise ValueError(
                    f"Output feature {output_feature} is not available.")

        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon

        self.train_batchsize = train_batchsize
        self.valid_batchsize = valid_batchsize

        self.input_features = input_features
        self.output_features = output_features
        self.bypass_features = [feature for feature in self.available_features
                                if feature not in self.output_features]

        # Get index of input, output and bypass features
        self.indices_input_features = [i for i in range(len(self.available_features))
                                       if self.available_features[i] in self.input_features]
        self.indices_output_features = [i for i in range(len(self.available_features))
                                        if self.available_features[i] in self.output_features]
        self.indices_bypass_features = [i for i in range(len(self.available_features))
                                        if i not in self.indices_output_features]
        self.scaling_factors_min: torch.Tensor = None
        self.scaling_factors_max: torch.Tensor = None
        self._preprocess_and_split_data()

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to save the state of the dataset."""
        return {
            'scaling_factors_min': self.scaling_factors_min,
            'scaling_factors_max': self.scaling_factors_max,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the dataset."""
        self.scaling_factors_min = state_dict['scaling_factors_min']
        self.scaling_factors_max = state_dict['scaling_factors_max']
        # Make sure to preprocess the data again
        self._preprocess_and_split_data()

    def _preprocess_and_split_data(self) -> None:
        """This method is used to preprocess and split the data into train, 
           validation and test sets.

        The data is normalised between 0 and 1 and then split into train,
        validation and test sets.
        Then the following variables are used:

        psi (torch.Tensor): Single input data sample at time step `t`.
        omega (torch.Tensor): Single target data sample at time step `t`.
        psi_batch (torch.Tensor): Batch of input data samples at time step `t`.
        omega_batch (torch.Tensor): Batch of target data samples at time step `t`.
        inputs_ `train/validation/test` (List[torch.Tensor]): list that collects the input data for each time step `t`.
        targets_ `train/validation/test` (List[torch.Tensor]): list that collects the target data for each time step `t`.
        """

        data = self._data[self.available_features].values
        # Normalise data between 0 and 1
        # Get the minimum and maximum values for each column
        # If we are transferring between different datasets, we use the loaded values
        self.scaling_factors_min = torch.from_numpy(data.min(
            axis=0)) if self.scaling_factors_min is None else self.scaling_factors_min
        self.scaling_factors_max = torch.from_numpy(data.max(
            axis=0)) if self.scaling_factors_max is None else self.scaling_factors_max

        # Convert to torch tensors and cast to float32
        data = torch.from_numpy(data).type(torch.float32)
        data_normalised = normalise(
            data, self.scaling_factors_min, self.scaling_factors_max)

        # Split data into train, validation and test sets
        self._inputs_train, self._targets_train = [], []
        self._inputs_validation, self._targets_validation = [], []
        self._inputs_test, self._targets_test = [], []

        # `t_i`` variable is the time step at which the controller has enough state
        # data such that the state data can be used.
        self.t_i = self.train_batchsize + self.valid_batchsize + \
            self.lookback_window + self.prediction_horizon - 1

        # `t_total` variable is the total number of time steps for which we have
        # the train, validation and test data.
        self.t_total = len(data) - self.prediction_horizon + 1

        for t in range(self.t_i, self.t_total):
            # Test data
            psi = torch.zeros(1, self.lookback_window,
                              len(self.available_features))
            omega = torch.zeros(1, self.prediction_horizon,
                                len(self.available_features))
            for j in range(len(self.available_features)):
                
                psi[0, :, j] = data_normalised[t - self.lookback_window:t, j]
                omega[0, :, j] = data_normalised[t:t +
                                                 self.prediction_horizon, j]
            self._inputs_test.append(psi)
            self._targets_test.append(omega)

            # Validation data
            psi_batch = torch.zeros(self.valid_batchsize,
                                    self.lookback_window, len(self.available_features))
            omega_batch = torch.zeros(self.valid_batchsize,
                                      self.prediction_horizon,
                                      len(self.available_features))

            for j in range(len(self.available_features)):
                for k in range(self.valid_batchsize):
                    # Single validation sample for each timestep
                    psi = data_normalised[t - self.lookback_window - self.prediction_horizon -
                                          k:t - self.prediction_horizon - k, j]
                    omega = data_normalised[t -
                                            self.prediction_horizon - k:t - k, j]
                    # Batch of validation samples for each timestep
                    psi_batch[k, :, j] = psi
                    omega_batch[k, :, j] = omega

            # Append to list of validation samples
            self._inputs_validation.append(psi_batch)
            self._targets_validation.append(omega_batch)

            # Training data
            psi_batch = torch.zeros(self.train_batchsize, self.lookback_window,
                                    len(self.available_features))
            omega_batch = torch.zeros(self.train_batchsize,
                                      self.prediction_horizon,
                                      len(self.available_features))
            for j in range(len(self.available_features)):
                for k in range(self.train_batchsize):
                    # Single training sample for each timestep
                    psi = data_normalised[t - self.valid_batchsize - self.lookback_window - self.prediction_horizon -
                                          k:t - self.valid_batchsize - self.prediction_horizon - k, j]
                    omega = data_normalised[t - self.valid_batchsize -
                                            self.prediction_horizon - k:t - self.valid_batchsize - k, j]
                    # Batch of validation samples for each timestep
                    psi_batch[k, :, j] = psi
                    omega_batch[k, :, j] = omega

            # Append to list of training samples
            self._inputs_train.append(psi_batch)
            self._targets_train.append(omega_batch)

    def get_profile(self) -> torch.Tensor:
        """This method is used to return the test profile."""
        data = self._data[self.available_features].values
        data = torch.from_numpy(data).type(torch.float32)
        profiles = data[self.t_i:self.t_total]
        return profiles

    def __len__(self) -> int:
        """This method is used to return the length of the dataset.

        This is the number of simulation steps:
        `len(self._inputs_test)) = len(self._inputs_validation) = len(self._inputs_train)`
        """
        return len(self._inputs_train)

    def __getitem__(self, time: int) -> Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor]]:
        """This method is used to return a sample from the dataset. 

        It contains the inputs and targets for the train, validation and test sets at a given timestep.

        To access the dataset samples use `__getitem__(i)[j][k]` where:
        - i = timestep
        - j = index for the train, validation or test set (0=train, 1=validation, 2=test)
        - k = index for the inputs or targets (0=inputs, 1=targets)

        Example:
        __getitem__(3)[0][1] = target samples, used for training at time step 3 
        """
        inputs_train = self._inputs_train[time]
        targets_train = self._targets_train[time]

        inputs_val = self._inputs_validation[time]
        targets_val = self._targets_validation[time]

        inputs_test = self._inputs_test[time]
        targets_test = self._targets_test[time]

        return (inputs_train, targets_train), (inputs_val, targets_val), (inputs_test, targets_test)

    @staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This method is used to add the model specific arguments to the parent parser."""
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument("--site_id", type=int,
                            default=12, help="The ID of the microgrid dataset.")
        parser.add_argument('--train_batchsize', type=int, default=1,
                            help='The number of samples used for each training step.')
        parser.add_argument('--valid_batchsize', type=int, default=1,
                            help='The number of samples used for each validation or test step.')
        parser.add_argument('--input_features',  nargs='+', default=['pv', 'load', 'Price'],
                            help='Features to use: Price, load, pv, state. Set to [] to define a prescient controller.')
        parser.add_argument('--output_features',  nargs='+', default=['pv', 'load', 'Price'],
                            help='Features to predict inside the controller: Price, load, pv. Set to [] to define a prescient controller.')
        parser.add_argument('--lookback_window', type=int, default=168,
                            help='Number of past time steps used for training.')
        parser.add_argument('--prediction_horizon', type=int, default=24,
                            help='The number of predicted future samples.')
        return parser
