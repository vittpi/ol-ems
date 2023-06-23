from typing import Any, Dict,  Tuple, Callable, Union
import os
import sys
from pathlib import Path
import logging
import pickle
import torch.nn as nn
import torch
import shutil
import importlib
from syne_tune import Reporter
from pytorch_lightning import Callback, Trainer, LightningModule
import time
import numpy as np
import datetime
from torch.utils.data import Dataset


class LinearScheduler():
    """This class implements a linear scheduler depending on a time step.

    Args:
        start_value (float): The value at the start of the scheduler.
        start_time_step (int): The step at which the scheduler starts.
        end_value (float): The value at the end of the scheduler.
        end_time_step (int): The step at which the scheduler ends.
    """

    def __init__(self, start_value: float, start_time_step: int,
                 end_value: float, end_time_step: int) -> None:
        self._start_value = start_value
        self._start_time_step = start_time_step
        self._end_value = end_value
        self._end_time_step = end_time_step
        assert self._start_time_step <= self._end_time_step, f"The start time step {self._start_time_step} needs to be smaller than the end time step {self._end_time_step}."
        self._step = 0

    def get_value(self) -> float:
        """This method is used to get the current value of the scheduler."""
        if self._step < self._start_time_step:
            return self._start_value
        elif self._step >= self._end_time_step:
            return self._end_value
        else:
            return self._start_value + (self._end_value - self._start_value) * (self._step - self._start_time_step) / (self._end_time_step - self._start_time_step)

    def reset(self) -> None:
        """This method is used to reset the scheduler."""
        self._step = 0

    def step(self) -> None:
        """This method is used to increment the step of the scheduler."""
        self._step += 1


class WeightDecay():
    """This class implements a weight decay applied to `model.parameters()`."""

    def __call__(self, model: nn.Module) -> torch.Tensor:
        """This method is used to apply the weight decay to the model parameters."""
        weight_decay_loss = 0.0
        for param in model.parameters():
            if param.requires_grad:
                weight_decay_loss += torch.sum(param ** 2) * 0.5
        return weight_decay_loss


def import_config_function_from_file(config_file: str) -> Callable:
    """This method is used to import a config function from a file.

    Note that the function name needs to be exactly `configuration_space`.
    """
    config_file_path = Path(config_file)
    config_file_dir = config_file_path.parent
    config_file_name = config_file_path.stem
    sys.path.append(str(config_file_dir))
    config_file = importlib.import_module(config_file_name)
    config_function = getattr(config_file, "configuration_space")
    return config_function


def config_logger(log_path: str) -> None:
    """This method is used to configure the logger."""
    log_path = os.path.join(log_path, "log.log")
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger("pytorch_lightning").addHandler(fh)


def get_current_time() -> str:
    """This method is used to get the current time to the millisecond."""
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")


def save_pickle(obj: Any, path: str) -> None:
    """This method is used to save a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """This method is used to load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model: nn.Module, model_path: str) -> None:
    """This method is used to save a model."""
    torch.save(model.state_dict(), model_path)


def load_model(model: nn.Module, model_path: str) -> nn.Module:
    """This method is used to load a model."""
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    pretrained_dict = dict(state_dict.items())
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict.keys()}
    # Perform check if everything is loaded properly
    for key, value in model_dict.items():
        if key not in pretrained_dict:
            raise ValueError(f"Missing key {key} in pretrained model")
        assert value.shape == pretrained_dict[
            key].shape, f"Shape mismatch for key {key}"
    # Check if there are any extra keys in the pretrained model
    for key, value in pretrained_dict.items():
        if key not in model_dict:
            raise ValueError(f"Extra key {key} in pretrained model")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def save_learner(learner: LightningModule, learner_path: str) -> None:
    """This method is used to save a learner as a pickle file."""
    save_pickle(learner.state_dict(), learner_path)


def save_dataset(dataset: Dataset, dataset_path: str) -> None:
    """This method is used to save a dataset as a pickle file."""
    save_pickle(dataset.state_dict(), dataset_path)


def save_learner_as_csv(learner: LightningModule, learner_path: str) -> None:
    """This method is used to save the state dict of a learner as a set of .csv files."""
    state_dict = learner.state_dict()
    # Create a directory with respect to the learner path
    Path(os.path.join(learner_path, "learner_state")).mkdir(
        parents=True, exist_ok=True)
    learner_state_folder = os.path.join(learner_path, "learner_state")

    def _recursively_save_value(key: str, value: Union[torch.Tensor, Dict], path: str) -> None:
        """This method is used to recursively save the state dict of a learner as a set of .csv files."""
        if isinstance(value, torch.Tensor):
            # Note that the values need to be flattened
            value = value.numpy().flatten()
            np.savetxt(os.path.join(path, "value.csv"),
                       value, delimiter=",")
        elif isinstance(value, dict):
            Path(os.path.join(path, key)).mkdir(
                parents=True, exist_ok=True)
            for sub_key, sub_value in value.items():
                _recursively_save_value(
                    sub_key, sub_value, os.path.join(path, key))
        else:
            raise ValueError(f"Unsupported value type {type(value)}")

    for key, value in state_dict.items():
        _recursively_save_value(key, value, learner_state_folder)


def load_learner(learner: LightningModule, learner_path: str) -> LightningModule:
    """This method is used to load a learner from a pickle file."""
    learner.load_state_dict(load_pickle(learner_path))
    return learner


def load_dataset(dataset: Dataset, dataset_path: str) -> Dataset:
    """This method is used to load a dataset from a pickle file."""
    dataset.load_state_dict(load_pickle(dataset_path))
    return dataset


def create_experiment_folder(experiment_path: str, src_folder: str) -> str:
    """This method is used to create the experiment folder and archite any code in the the src_folder."""
    # Check if the experiment folder exists
    counter = 0
    original_experiment_path = experiment_path
    while os.path.exists(experiment_path):
        # Wait randomly up to 5 seconds to avoid deadlock
        time.sleep(torch.randint(
            0, 5, (1,), generator=torch.Generator()).item())
        experiment_path = original_experiment_path + f"-{counter}"
        counter += 1

    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    script_folder = os.path.join(experiment_path, 'scripts')
    os.mkdir(script_folder)

    # Create a directory with respect to the script path
    # and copy all the files in the src folder while preserving the folder structure relative the to `src_folder`
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_path = os.path.relpath(file_path, src_folder)
                file_path = os.path.join(script_folder, file_path)
                Path(os.path.dirname(file_path)).mkdir(
                    parents=True, exist_ok=True)
                shutil.copy(os.path.join(root, file), file_path)
    return experiment_path


class ReporterCallback(Callback):
    """This callback reports the all the metrics back to the Syne-Tune Reporter every training step.

    Args:
        reporter (Reporter): The Syne-Tune Reporter.
    """

    def __init__(self, reporter: Reporter) -> None:
        super().__init__()
        self._reporter = reporter

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int,
                           dataloader_idx: int) -> None:
        """This method reports all the recorded metrics back to the Syne-Tune Reporter.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (LightningModule): The PyTorch Lightning Module.
            outputs (Any): The outputs passed by the training step.
            batch (Any): The batch.
            batch_idx (int): The batch index.
            dataloader_idx (int): The dataloader index.
        """
        # If we are in the sanity check, we do not report anything.
        if trainer.sanity_checking:
            return
        logged_metrics = {}
        for key, value in trainer.logged_metrics.items():
            if "train" not in key and "val" not in key and "test" not in key:
                continue
            key = key.replace("/", "_")
            logged_metrics[key] = value.item()
        self._reporter(step=pl_module.time_step,
                       **logged_metrics,
                       epoch=pl_module.time_step+1)


def store_metrics(results: Dict[str, Any], metrics: nn.ModuleDict, prefix: str) -> None:
    """This method is used to store the metrics.

    Args:
        results (Dict[str, Any]): The dictionary where the metrics are stored.
        metrics (nn.ModuleDict): The metrics.
        prefix (str): The prefix to be used for the metrics.
    """
    results[prefix] = {}
    for metric_name, metric_value in metrics.items():
        results[prefix][metric_name] = metric_value.compute().item()


def results_file(experiment_path: str) -> str:
    """This method is used to create the results file."""
    return os.path.join(experiment_path, "results.pickle")


def model_file(experiment_path: str) -> str:
    """This method is used to create the model file."""
    return os.path.join(experiment_path, "model.pth")


def learner_file(experiment_path: str) -> str:
    """This method is used to create the learner file."""
    return os.path.join(experiment_path, "learner_state.pickle")


def dataset_file(experiment_path: str) -> str:
    """This method is used to create the dataset file."""
    return os.path.join(experiment_path, "dataset_state.pickle")


def args_file(experiment_path: str) -> str:
    """This method is used to create the args file."""
    return os.path.join(experiment_path, "args.pickle")


def denormalise(data: torch.Tensor, minimum: Union[np.array, torch.Tensor], maximum: Union[np.array, torch.Tensor]) -> torch.Tensor:
    """This function is used to denormalise the data."""
    assert minimum.shape == maximum.shape, f"Minimum and maximum shapes do not match: {minimum.shape} != {maximum.shape}."
    assert data.shape[-1] == minimum.shape[-1], f"Data and minimum shapes do not match: {data.shape[-1]} != {minimum.shape[-1]}."
    assert data.shape[-1] == maximum.shape[-1], f"Data and maximum shapes do not match: {data.shape[-1]} != {maximum.shape[-1]}."
    maximum, minimum = maximum.to(data.device), minimum.to(data.device)
    return data * (maximum - minimum) + minimum


def normalise(data: torch.Tensor, minimum: Union[np.array, torch.Tensor], maximum: Union[np.array, torch.Tensor]) -> torch.Tensor:
    """This function is used to normalise the data."""
    assert minimum.shape == maximum.shape, f"Minimum and maximum shapes do not match: {minimum.shape} != {maximum.shape}."
    assert data.shape[-1] == minimum.shape[-1], f"Data and minimum shapes do not match: {data.shape[-1]} != {minimum.shape[-1]}."
    assert data.shape[-1] == maximum.shape[-1], f"Data and maximum shapes do not match: {data.shape[-1]} != {maximum.shape[-1]}."
    maximum, minimum = maximum.to(data.device), minimum.to(data.device)
    return (data - minimum) / (maximum - minimum)
