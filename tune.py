from typing import Dict, Any

from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import baselines_dict
from syne_tune.backend import LocalBackend
from syne_tune import Tuner, StoppingCriterion
import tabulate
from argparse import ArgumentParser
import pytorch_lightning as pl
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging
import random

import src.utils as utils
import src.plots as plots

logging = logging.getLogger('pytorch_lightning')


def tune(args: ArgumentParser) -> None:
    # Set seed
    # Create experiment structure
    optimizer_name = args.optimizer.replace(" ", "-")
    experiment_name = f"{utils.get_current_time()}-tune-{optimizer_name}"

    # Create experiment directory
    experiment_path = os.path.join("experiments", experiment_name)
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    utils.config_logger(experiment_path)

    logging.info("Beginning tuning: %s", experiment_name)
    logging.info("Arguments: %s", args)

    pl.seed_everything(args.seed)
    config_space = utils.import_config_function_from_file(args.config_file)()
    config_space = {**config_space, **args.additional_arguments}

    logging.info("Config space: %s", config_space)
    scheduler_kwargs = {"config_space": config_space, "metric": args.optimization_metric,
                        "max_t": 1, "random_seed": args.seed, "mode": "min"}
    if args.optimizer == "ASHA":
        scheduler_kwargs["resource_attr"] = "epoch"
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point="main.py"),
        scheduler=baselines_dict[args.optimizer](**scheduler_kwargs),
        stop_criterion=StoppingCriterion(
            max_wallclock_time=args.max_wallclock_time, max_num_trials_started=args.max_num_trials_started),
        n_workers=args.n_workers,
        tuner_name=experiment_name,
    )

    tuner.run()
    tuning_experiment = load_experiment(tuner.name)
    tuning_experiment.plot()
    plt.grid()
    plt.savefig(os.path.join(experiment_path, "tuning.png"),
                bbox_inches="tight")
    plt.close()
    plt.clf()
    best_config = tuning_experiment.best_config()
    logging.info("Best configuration: %s", best_config)
    logging.info("Best configuration command: %s", best_config_to_command_arguments(
        best_config))
    utils.save_pickle(best_config, os.path.join(
        experiment_path, "best_configuration.pkl"))
    # Write it also as a text file
    with open(os.path.join(experiment_path, "best_configuration.txt"), "w") as f:
        f.write(str(best_config))

    results_df = tuning_experiment.results
    #results_df.to_csv(os.path.join(experiment_path, "results.csv"))
    #logging.info("Results dataframe: %s", tabulate.tabulate(
    #    results_df, headers="keys", tablefmt="psql"))
    plots.plot_different_runs_and_metrics(results_df, experiment_path)


def best_config_to_command_arguments(best_config: Dict[str, Any]) -> str:
    """Convert the best config to command arguments."""
    command = ""
    for key, value in best_config.items():
        if "config_" in key:
            key = key.replace("config_", "")
            if isinstance(value, list):
                for v in value:
                    command += f"--{key} {v} "
            else:
                command += f"--{key} {value} "

    return command


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config_file', type=str, default=None,
                        help='The config file to be used for hyperparameter tuning.')
    parser.add_argument('--optimization_metric', type=str, default="val_loss_mse",
                        help='The metric to be used for hyperparameter optimization.')
    parser.add_argument('--additional_arguments', type=json.loads, default={},
                        help='Additional arguments to be used for hyperparameter tuning.')

    parser.add_argument_group("Experiment")
    parser.add_argument('--seed', type=int, default=42,
                        help='The seed to be used for training.')

    parser.add_argument_group("Tuning")
    parser.add_argument('--max_wallclock_time', type=int, default=3*3600,
                        help='The maximum wallclock time to be used for hyperparameter tuning.')
    parser.add_argument('--max_num_trials_started', type=int, default=500,
                        help='The maximum number of trials to be started for hyperparameter tuning.')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='The number of workers to be used for hyperparameter tuning.')
    parser.add_argument('--optimizer', type=str, choices=list(baselines_dict.keys()),
                        default='Random Search', help='The optimizer to be used for hyperparameter tuning.')

    args = parser.parse_known_args()[0]
    args = parser.parse_args()

    tune(args)
