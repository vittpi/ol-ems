from typing import Tuple, Dict, Any
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchmetrics
import argparse

from src.plants.plants import MicroGridSystem
from src.models.controller import Controller
from src.dataloader.csv_dataset import CSVDataset
from src.utils import denormalise, LinearScheduler, WeightDecay

import logging
logging = logging.getLogger('pytorch_lightning')

# TODO: Define the tendency for total_cost metric
METRICS_TENDENCY = {"train_loss_mse": "min", "val_loss_mse": "min", "test_loss_mse": "min", "train_loss_weight_decay": "min", "train_loss": "min",
                    "train_loss_task": "min", "val_loss_task": "min", "test_loss_task": "min",
                    "train_loss_total_cost": "min", "val_loss_total_cost": "min", "test_loss_total_cost": "min"}


class OnlineLearner(pl.LightningModule):
    """This class is used to group all modules together and perform the training and evaluation of the complete model.

    Args:
        plant (MicroGridSystem): The plant model to be used.
        controller (nn.Module): The controller to be trained.
        regression_loss (nn.Module): The loss function to be used for regression.
        total_cost_loss (nn.Module): The loss function to be used for the total cost.
        dataset (CSVDataset): The dataset to be used for training and evaluation.

        learning_rate (float): The learning rate to be used for training.
        weight_decay_start_weight (float): The weight of the weight decay at the start of training.
        weight_decay_end_weight (float): The weight of the weight decay at the end of training.
        weight_decay_start_time_step (int): The time step at which to start the weight schedule of the weight decay.
        weight_decay_end_time_step (int): The time step at which to end the weight schedule of the weight decay.

        swa_gamma (float): The SWA gamma factor that multiplies the current average when computing the moving average.
        swa_replace_frequency (int): How often to replace the network parameters with the SWA copy.
        swa_start_time_step (int): The time step at which to start using SWA.
        swa_end_time_step (int): The time step at which to stop using SWA.

        mse_start_weight (float): The weight of the MSE loss at the start of training.
        mse_exp_decay (float): The exponential decay factor of the MSE prediction horizon loss weight.
        mse_end_weight (float): The weight of the MSE loss at the end of training.
        mse_start_time_step (int): The time step at which to start the weight schedule of the MSE loss.
        mse_end_time_step (int): The time step at which to end the weight schedule of the MSE loss.

        task_start_weight (float): The weight of the task loss at the start of training.
        task_end_weight (float): The weight of the task loss at the end of training.
        task_start_time_step (int): The time step at which to start the weight schedule of the task loss.
        task_end_time_step (int): The time step at which to end the weight schedule of the task loss.
        task_window (int): The window size for the task loss for validation and testing.
    """

    def __init__(self,
                 plant: MicroGridSystem,
                 controller: Controller,
                 regression_loss: nn.Module,
                 total_cost_loss: nn.Module,
                 dataset: CSVDataset,

                 learning_rate: float = 1e-3,
                 weight_decay_start_weight: float = 1e-4,
                 weight_decay_end_weight: float = 1e-4,
                 weight_decay_start_time_step: int = 0,
                 weight_decay_end_time_step: int = 1e8,

                 swa_gamma: float = 0.99,
                 swa_replace_frequency: int = 5,
                 swa_start_time_step: int = 0,
                 swa_end_time_step: int = 1e8,

                 mse_start_weight: float = 1.0,
                 mse_exp_decay: float = 0.8,
                 mse_end_weight: float = 1.0,
                 mse_start_time_step: int = 0,
                 mse_end_time_step: int = 1e8,

                 task_start_weight: float = 1.0,
                 task_end_weight: float = 1.0,
                 task_start_time_step: int = 0,
                 task_end_time_step: int = 1e8,
                 task_window: int = 168,
                 ) -> None:
        super(OnlineLearner, self).__init__()
        self.save_hyperparameters(
            ignore=["plant", "controller", "loss", "dataset"])

        self._plant = plant
        self._controller = controller
        self._regression_loss = regression_loss
        self._total_cost_loss = total_cost_loss

        self._input_features = dataset.input_features
        self._indices_input_features = dataset.indices_input_features
        self._output_features = dataset.output_features
        self._indices_output_features = dataset.indices_output_features
        self._indices_bypass_features = dataset.indices_bypass_features
        self._available_features = dataset.available_features
        self._scaling_factors_min = dataset.scaling_factors_min
        self._scaling_factors_max = dataset.scaling_factors_max
        self._prediction_horizon = dataset.prediction_horizon
        self._lookback_window = dataset.lookback_window
        self._price_profile = dataset.get_profile()[:, 2]

        self._output_dim = len(dataset.output_features)
        self._train_batchsize = dataset.train_batchsize
        self._valid_batchsize = dataset.valid_batchsize

        self._t_bar = dataset.t_i + 1

        self._create_metrics()
        self._create_state_inputs_containers()
        self._create_prediction_containers()

        # Internal timer to keep track of the simulation time
        self.time_step = 0

        # Initialize the weight schedulers for the losses
        self._mse_weight_scheduler = LinearScheduler(
            start_value=self.hparams.mse_start_weight,
            end_value=self.hparams.mse_end_weight,
            start_time_step=self.hparams.mse_start_time_step,
            end_time_step=self.hparams.mse_end_time_step,
        )

        self._task_weight_scheduler = LinearScheduler(
            start_value=self.hparams.task_start_weight,
            end_value=self.hparams.task_end_weight,
            start_time_step=self.hparams.task_start_time_step,
            end_time_step=self.hparams.task_end_time_step,
        )

        # Initialize the weight schedulers for the weight decay
        self._weight_decay_scheduler = LinearScheduler(
            start_value=self.hparams.weight_decay_start_weight,
            end_value=self.hparams.weight_decay_end_weight,
            start_time_step=self.hparams.weight_decay_start_time_step,
            end_time_step=self.hparams.weight_decay_end_time_step,
        )
        self._weight_decay = WeightDecay()

        # Create a copy of the controller to be used for the SWA
        self._swa_model_copy = [p.clone().detach()
                                for p in self._controller.parameters()]

    @torch.no_grad()
    def _update_swa_model_copy(self) -> None:
        """This method is used to update the SWA model copy.

        It is a weighted average of the model copy and the current model parameters.
        Such that `new_model_copy = swa_gamma * model_copy + (1 - swa_gamma) * current_model`
        """
        for param, swa_param in zip(self._controller.parameters(), self._swa_model_copy):
            swa_param.data.mul_(self.hparams.swa_gamma).add_(
                param.data.detach().to(swa_param.data.device), alpha=1.0 - self.hparams.swa_gamma)

    @torch.no_grad()
    def _replace_current_model_with_swa_model_copy(self) -> None:
        """This method is used to replace the current model with the SWA model copy."""
        for param, swa_param in zip(self._controller.parameters(), self._swa_model_copy):
            param.data.copy_(swa_param.data)

    def _create_state_inputs_containers(self) -> None:
        """This method is used to create the containers for the system state and
            the imputs computed by the controller.

        `self.state`: This container is used to store the system state. At the
            end of the simulation, the shape of this container is `(simulation_steps+1, num_states)`.

        `self.controller_outputs`: This container is used to store the outputs computed by the
            controller, the shape of this containeris `(simulation_steps, num_inputs)`.
        """
        self.state = self._plant.get_initial_state()
        self.controller_outputs = torch.empty(
            0, self._plant.get_input_dimension()+1, dtype=torch.float64)

    def state_dict(self) -> Dict[str, Any]:
        """This method is used to get the state of the learner."""
        return {
            "state": self.state.clone().detach().cpu(),
            "decisions":
                {
                "train": {key: value.clone().detach().cpu() for key, value in self.decisions["train"].items()},
                "val": {key: value.clone().detach().cpu() for key, value in self.decisions["val"].items()},
                "test": {key: value.clone().detach().cpu() for key, value in self.decisions["test"].items()}
            },
            "predictions": {key: value.clone().detach().cpu() for key, value in self.predictions.items()},
            "targets": {key: value.clone().detach().cpu() for key, value in self.targets.items()},
            "controller_outputs": self.controller_outputs.clone().detach().cpu(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """This method is used to load the state of the learner."""
        self.state = state_dict["state"]
        self.decisions = state_dict["decisions"]
        self.predictions = state_dict["predictions"]
        self.targets = state_dict["targets"]
        self.controller_outputs = state_dict["controller_outputs"]

    def _create_prediction_containers(self) -> None:
        """This method is used to create the containers for the predictions."""
        self.decisions = {}
        self.predictions = {}
        self.targets = {}
        for data_split in ["train", "val", "test"]:
            self.decisions[data_split] = {feature: torch.empty((0, self._prediction_horizon))
                                          for feature in ['Ps_in', 'Ps_out', 'Pl', 'Pr', 'Pg', 'eps']}
            self.predictions[data_split] = torch.empty((0,
                                                        self._prediction_horizon,
                                                        self._output_dim))
            self.targets[data_split] = torch.empty((0, self._prediction_horizon,
                                                    self._output_dim))

    @torch.no_grad()
    def _append_predicition_containers(self, data_split: str, 
                                       predictions: Tuple[Tuple[torch.Tensor, ...]], 
                                       targets: torch.Tensor) -> None:
        """This method is used to append the predictions to the containers.
        
        Args:
            data_split (str): The data split to which the predictions belong (train/validation/test).
            predictions (Tuple[Tuple[torch.Tensor, ...]]): The predictions to be appended.
            targets (torch.Tensor): The targets to be appended.
        """

        predictions_hat, decision_variables = predictions
        # Save predictions
        self.predictions[data_split] = torch.cat(
            (self.predictions[data_split], predictions_hat.detach().cpu()), dim=0)
        self.targets[data_split] = torch.cat(
            (self.targets[data_split], targets.detach().cpu()), dim=0)

        # Save decision variables
        (Ps_in, Ps_out, Pl, Pr, eps) = decision_variables
        for feature, value in zip(['Ps_in', 'Ps_out', 'Pl', 'Pr', 'eps'], [Ps_in, Ps_out, Pl, Pr, eps]):
            self.decisions[data_split][feature] = torch.cat(
                (self.decisions[data_split][feature], value.detach().cpu()), dim=0)

        self.decisions[data_split]['Pg'] = torch.cat((self.decisions[data_split]['Pg'],
                                                      self._plant.compute_Pg(Ps_in=Ps_in, Ps_out=Ps_out, Pr=Pr, Pl=Pl).detach().cpu()),
                                                      dim=0)

    def _create_metrics(self) -> None:
        """This method is used to create the metrics to be used for training, validation and testing."""
        self.metrics = {
            "train": nn.ModuleDict({
                "loss_mse": torchmetrics.MeanMetric(),
                "loss_weight_decay": torchmetrics.MeanMetric(),
                "loss": torchmetrics.MeanMetric(),
            }),
            "val": nn.ModuleDict({
                "loss_mse": torchmetrics.MeanMetric(),

            }),
            "test": nn.ModuleDict({
                "loss_mse": torchmetrics.MeanMetric(),
            }),
        }

    def _get_state(self, device: torch.device
                   ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                              Tuple[torch.Tensor, torch.Tensor],
                              Tuple[torch.Tensor]]:
        """This method is used to get the state of the system to be used for  training, validation and testing."""
        input_state_test = self.state[0] * torch.ones(1, self._lookback_window, self._plant.get_state_dimension(),
                                                      device=device, dtype=torch.float64)
        input_state_validation = self.state[0] * torch.ones(self._valid_batchsize, self._lookback_window,
                                                            self._plant.get_state_dimension(), device=device, dtype=torch.float64)
        target_state_validation = self.state[0] * torch.ones(self._valid_batchsize, self._prediction_horizon,
                                                             self._plant.get_state_dimension(), device=device, dtype=torch.float64)
        input_state_training = self.state[0] * torch.ones(self._train_batchsize, self._lookback_window,
                                                          self._plant.get_state_dimension(), device=device, dtype=torch.float64)
        target_state_training = self.state[0] * torch.ones(self._train_batchsize, self._prediction_horizon, self._plant.get_state_dimension(),
                                                           device=device, dtype=torch.float64)
        if self.time_step >= self._t_bar:
            # Initialise the tensors
            input_state_test.zero_()
            input_state_validation.zero_()
            target_state_validation.zero_()
            input_state_training.zero_()
            target_state_training.zero_()

            # The last lookback samples are used for test (i.e. computing the control input at time t)
            for j in range(self._plant.get_state_dimension()):
                input_state_test[0, :, j] = self.state[self.time_step -
                                                       self._lookback_window+1: self.time_step+1, j]

            # `target_state_test`` is not known at time t since it is the state at
            # time t+1, t+2, ..., t+prediction_horizon
            # Past samples are used for training and validation
            # The past state sampels are known at time t

            # Validation data
            for j in range(self._plant.get_state_dimension()):
                for k in range(self._valid_batchsize):
                    input_state_validation[k, :, j] = \
                        self.state[self.time_step-self._lookback_window-self._prediction_horizon -
                                   k+1: self.time_step-self._prediction_horizon-k+1, j]
                    target_state_validation[k, :, j] = \
                        self.state[self.time_step-self._prediction_horizon -
                                   k+1: self.time_step-k+1, j]

            # Training data
            for j in range(self._plant.get_state_dimension()):
                for k in range(self._train_batchsize):
                    input_state_training[k, :, j] = \
                        self.state[self.time_step-self._lookback_window-self._prediction_horizon-self._valid_batchsize-k+1:
                                   self.time_step-self._valid_batchsize-self._prediction_horizon-k+1, j]
                    target_state_training[k, :, j] = \
                        self.state[self.time_step-self._valid_batchsize-self._prediction_horizon-k+1:
                                   self.time_step-self._valid_batchsize-k+1, j]
        return ((input_state_training, target_state_training),
                (input_state_validation, target_state_validation),
                (input_state_test))

    def _split_and_squeeze(self, pair: Tuple[torch.Tensor, torch.Tensor]
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to split and squeeze the input tensor.
        
        Args:
            pair (Tuple[torch.Tensor, torch.Tensor]): The input tensor to be split and squeezed.
        """
        inputs, targets = pair
        inputs, targets = inputs.squeeze(0), targets.squeeze(0)
        return inputs, targets

    def _features_selection(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
                                               Tuple[torch.Tensor, torch.Tensor],
                                               Tuple[torch.Tensor, torch.Tensor]]
                            ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """This method is used to select the features from the input tensor to
            be used as input to the controller. The other features bypass the neural network
            and go straight to the optimisation layer.
            
        Args:
            batch (Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]): The batch of data.
        """

        # Unpack the data and divide them into training, validation and testing sets
        inputs_training, targets_training = self._split_and_squeeze(batch[0])
        inputs_validation, targets_validation = self._split_and_squeeze(batch[1])
        inputs_test, targets_test = self._split_and_squeeze(batch[2])

        batch = []
        for inputs, targets in zip([inputs_training, inputs_validation, inputs_test],
                                   [targets_training, targets_validation, targets_test]):

            inputs = inputs[:, :, self._indices_input_features]
            bypass = targets[:, :, self._indices_bypass_features]
            targets = targets[:, :, self._indices_output_features]
            batch.append((inputs, targets, bypass))

        return batch

    def _plant_simulation(self, Ps_in: torch.Tensor,
                          Ps_out: torch.Tensor,
                          batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, torch.Tensor]]
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method is used to simulate the plant with the computed input and
           the current values of the `pv` and `load`.

        Args:
            Ps_in (torch.Tensor): The input power computed by the controller.
            Ps_out (torch.Tensor): The output power computed by the controller.
            batch: The batch of data.
        """
        # The first computed input is applied to the system and we discard the others
        current_input_in = Ps_in[:, 0:1]
        current_input_out = Ps_out[:, 0:1]
        current_input = torch.cat((current_input_in, current_input_out), dim=0)

        # State at current time step
        current_state = self.state[self.time_step:self.time_step+1, :]
        # Compute next state
        next_state = self._plant.plant_simulation(
            current_state, current_input)
        # Avoid negative states due to solver tolerance
        next_state = torch.clamp(next_state, min=self._plant.charge_min_max[0],
                                 max=self._plant.charge_min_max[1])

        # [2]=test, [1]=targets, [0, 0] = first sample of pv and load
        test_targets = batch[2][1][0, 0]
        profiles = denormalise(test_targets, 
            self._scaling_factors_min.unsqueeze(0), self._scaling_factors_max.unsqueeze(0))

        pv, load = profiles[0:1, 0:1], profiles[0:1, 1:2]

        # Compute Pg with the actual values of pv and load
        Pg = self._plant.compute_Pg(Ps_in=current_input_in ,Ps_out=current_input_out , Pr=pv, Pl=load).detach()
        return torch.cat((current_input_in, current_input_out, Pg), dim=1), next_state

    def on_fit_start(self) -> None:
        """This method is called when the training starts."""
        # Move the variables to correct device
        self.state = self.state.to(self.device)
        return super().on_fit_start()

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, torch.Tensor]],
                      batch_idx: int) -> torch.Tensor:
        """This method is used to perform a single training, validation and testing step.
        
        Args:
            batch (Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]): The batch of data.
            batch_idx (int): The index of the batch.
        """
        super(OnlineLearner, self).training_step(batch, batch_idx)
        self._controller.train()

        # Unpack the data and divide them into training, validation and testing sets
        # The bypass tensor is used to pass the target features directly to the optimiser
        (inputs_training, targets_training, bypass_training), \
            (inputs_validation, targets_validation, bypass_validation), \
            (inputs_test, targets_test, bypass_test) = self._features_selection(batch)

        # Get the state data
        (input_state_training, target_state_training), \
            (input_state_validation, target_state_validation), \
            (input_state_test) = self._get_state(inputs_training.device)

        training_predictions = self._controller(inputs=inputs_training,
                                                bypass=bypass_training,
                                                initial_state=input_state_training[:, -1])
        targets_hat_train, _ = training_predictions
        weight_per_prediction_horizon = self._regression_loss.get_weight(
            self._prediction_horizon, self.hparams.mse_exp_decay, self.device)
        # Compute the training loss
        gradient_loss = 0.0
        regression_loss_training = self._regression_loss(
            targets_hat_train, targets_training, weight=weight_per_prediction_horizon)
        gradient_loss += regression_loss_training * \
            self._mse_weight_scheduler.get_value()
        weight_decay = self._weight_decay(self._controller)
        gradient_loss += weight_decay*self._weight_decay_scheduler.get_value()
        # TODO: task loss not yet implemented
        task_loss_training = torch.zeros_like(regression_loss_training)
        gradient_loss += task_loss_training*self._task_weight_scheduler.get_value()

        if self.time_step > self.hparams.task_window:
            total_cost_loss_training = self._total_cost_loss(
                price=self._price_profile[self.time_step -
                                          self.hparams.task_window:self.time_step],
                Pg=self.controller_outputs[self.time_step-self.hparams.task_window:self.time_step, 1])
        else:
            total_cost_loss_training = 1e8

        self.metrics["train"]["loss_mse"].update(
            regression_loss_training.item())
        self.metrics["train"]["loss_weight_decay"].update(
            weight_decay.item())
        self.metrics["train"]["loss"].update(
            gradient_loss.item())
        # self.metrics["train"]["loss_task"].update(task_loss_training.item())
        # self.metrics["train"]["loss_total_cost"].update(total_cost_loss_training)

        self._append_predicition_containers(
            "train", training_predictions, targets_training)

        with torch.no_grad():
            self._controller.eval()
            valid_predictions = self._controller(inputs=inputs_validation,
                                                 bypass=bypass_validation,
                                                 initial_state=input_state_validation[:, -1])
            targets_hat_valid, _ = valid_predictions
            # Compute the validation loss
            regression_loss_validation = self._regression_loss(
                targets_hat_valid, targets_validation)
            # TODO: task loss not yet implemented
            task_loss_validation = torch.zeros_like(regression_loss_validation)
            self.metrics["val"]["loss_mse"].update(
                regression_loss_validation.item())
            # self.metrics["val"]["loss_task"].update(total_cost_loss_training)
            # self.metrics["val"]["loss_total_cost"].update(total_cost_loss_training)
            self._append_predicition_containers(
                "val", valid_predictions, targets_validation)

            # Test
            test_predictions = self._controller(inputs=inputs_test,
                                                bypass=bypass_test,
                                                initial_state=self.state[self.time_step:self.time_step+1, 0:1])
            targets_hat_test, decision_variables = test_predictions
            # Compute the test loss
            regression_loss_test = self._regression_loss(
                targets_hat_test, targets_test)
            # TODO: task loss not yet implemented
            task_loss_test = torch.zeros_like(regression_loss_test) 
            self.metrics["test"]["loss_mse"].update(
                regression_loss_test.item())
            # self.metrics["test"]["loss_task"].update(task_loss_test.item())
            # self.metrics["test"]["loss_total_cost"].update(total_cost_loss_training)

            # Save predictions
            self._append_predicition_containers(
                "test", test_predictions, targets_test)

            # Save decision variables
            (Ps_in_test, Ps_out_test, _, _, _) = decision_variables

            # MPC controller implementation and plant simulation
            current_input, next_state = self._plant_simulation(Ps_in = Ps_in_test,
                                                               Ps_out=Ps_out_test,
                                                               batch = batch)

            # Save state and input
            self.controller_outputs = torch.cat(
                (self.controller_outputs, current_input.detach().cpu()), dim=0)
            self.state = torch.cat((self.state, next_state.detach()), dim=0)

        # Increase timer
        self.time_step += 1

        # Log metrics after each batch
        self._log_metrics("train", self.metrics["train"])
        self.log("val/loss_total_cost", total_cost_loss_training,
                 on_step=True, on_epoch=False)
        self._log_metrics("val", self.metrics["val"])
        self._log_metrics("test", self.metrics["test"])

        return gradient_loss

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Any, batch_idx: int) -> None:
        """This method is used to perform the operations at the end of each training, validation and testing batch."""
        self._update_swa_model_copy()
        if self.time_step % self.hparams.swa_replace_frequency == 0 \
                and self.time_step > 0 and self.hparams.swa_replace_frequency > 0 \
                and self.time_step >= self.hparams.swa_start_time_step \
                and self.time_step <= self.hparams.swa_end_time_step:
            self._replace_current_model_with_swa_model_copy()

        self.log("mse_weight", self._mse_weight_scheduler.get_value(),
                 on_step=True, on_epoch=False)
        self.log("task_weight", self._task_weight_scheduler.get_value(),
                 on_step=True, on_epoch=False)
        self.log("weight_decay", self._weight_decay_scheduler.get_value(),
                 on_step=True, on_epoch=False)

        self._mse_weight_scheduler.step()
        self._task_weight_scheduler.step()
        self._weight_decay_scheduler.step()

        super().on_train_batch_end(outputs, batch, batch_idx)

    def _log_metrics(self, prefix: str, metrics: nn.ModuleDict) -> None:
        """This method is used to log the metrics.

        Args:
            prefix (str): The prefix to be used for the logged metrics.
            metrics (nn.ModuleDict): The metrics to be logged.
        """
        for metric_name, metric in metrics.items():
            self.log(f"{prefix}/{metric_name}", metric.compute(),
                     on_step=True, on_epoch=False)

    def _reset_metrics(self, prefix: str) -> None:
        """This method is used to reset the metrics.

        The metrics are not reset at the end of training, validation and testing, because they are logged externally.

        Args:
            prefix (str): The prefix of the metrics to be reset.
        """
        if self.current_epoch < self.trainer.max_epochs - 1:
            for metric in self.metrics[prefix].values():
                metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """This method is used to configure the optimizer to be used for training."""
        return torch.optim.Adam(self._controller.parameters(), lr=self.hparams.learning_rate, weight_decay=0.0)

    @ staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """This method is used to add learner specific arguments to the parent parser."""
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-2,
                            help="The learning rate to be used for training.")
        
        parser.add_argument("--weight_decay_start_weight", type=float, default=0.0001,
                            help="The weight decay to be used at the start of training.")
        parser.add_argument("--weight_decay_end_weight", type=float, default=0.0001,
                            help="The weight decay to be used at the end of training.")
        parser.add_argument("--weight_decay_start_time_step", type=int, default=0,
                            help="The starting time step for the weight decay.")
        parser.add_argument("--weight_decay_end_time_step", type=int, default=int(1e8),
                            help="The ending time step for the weight decay.")

        parser.add_argument("--swa_gamma", type=float, default=0.5,
                            help="The SWA gamma factor that multiplies the current average when computing the moving average.")
        parser.add_argument("--swa_start_time_step", type=int, default=0,
                            help="The starting time step for the SWA.")
        parser.add_argument("--swa_end_time_step", type=int, default=int(1e8),
                            help="The ending time step for the SWA.")
        parser.add_argument("--swa_replace_frequency", type=int, default=5,
                            help="How often to replace the network parameters with the SWA copy.")

        parser.add_argument("--mse_start_weight", type=float, default=1.0,
                            help="The starting weight of the MSE loss.")
        parser.add_argument("--mse_exp_decay", type=float, default=0.8,
                            help="The exponential decay weighting for the individual samples.")
        parser.add_argument("--mse_end_weight", type=float, default=1.0,
                            help="The ending weight of the MSE loss.")
        parser.add_argument("--mse_start_time_step", type=int, default=0,
                            help="The starting time step for the MSE loss weight.")
        parser.add_argument("--mse_end_time_step", type=int, default=int(1e8),
                            help="The ending time step for the MSE loss weight.")

        parser.add_argument("--task_start_weight", type=float, default=1.0,
                            help="The starting weight of the task loss.")
        parser.add_argument("--task_end_weight", type=float, default=1.0,
                            help="The ending weight of the task loss.")
        parser.add_argument("--task_start_time_step", type=int, default=0,
                            help="The starting time step for the task loss weight.")
        parser.add_argument("--task_end_time_step", type=int, default=int(1e8),
                            help="The ending time step for the task loss weight.")
        parser.add_argument("--task_window", type=int, default=168,
                            help="The window size for the task loss for validation and testing.")

        return parser
