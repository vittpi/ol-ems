from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import logging

import src.utils as utils
import src.plots as plots
from src.training_and_evaluation.loss import MSELoss, TotalCostLoss
from src.plants.plants import MicroGridSystem
from src.models.controller import Controller, OptimisationLayer
from src.dataloader.csv_dataset import CSVDataset
from src.training_and_evaluation.learner import OnlineLearner
from syne_tune import Reporter

import logging
logging = logging.getLogger('pytorch_lightning')

def main(args: ArgumentParser) -> None:
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create experiment structure
    if args.experiment_name is None:
        experiment_name = utils.get_current_time()
    else:
        experiment_name = args.experiment_name
    experiment_path = os.path.join(args.save_dir, experiment_name)
    experiment_path = utils.create_experiment_folder(experiment_path, "./src")

    # Set the logger
    utils.config_logger(experiment_path)
    logging.info("Beginning experiment: %s", experiment_name)
    logging.info("Arguments: %s", args)

    # Save the arguments
    utils.save_pickle(args, utils.args_file(experiment_path))

    # Load data
    dataset = CSVDataset(site_id=args.site_id,
                         lookback_window=args.lookback_window,
                         prediction_horizon=args.prediction_horizon,
                         train_batchsize=args.train_batchsize,
                         valid_batchsize=args.valid_batchsize,
                         input_features=args.input_features,
                         output_features=args.output_features)
    if args.load_dir is not None:
        logging.info(
            "Loading dataset scaling constants from %s", args.load_dir)
        utils.load_dataset(dataset, utils.dataset_file(args.load_dir))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Create the plant model
    microgrid = MicroGridSystem(site_id=args.site_id,
                                sample_time=args.sample_time,
                                self_discharge=args.self_discharge,
                                initial_state=args.initial_state,
                                grid_power_min_max=args.grid_power_min_max,)

    # Create the NN and MPC optimiser
    optimisation_layer = OptimisationLayer(prediction_horizon=args.prediction_horizon,
                                           A_matrix=microgrid.get_model_matrices()[0],
                                           B_matrix=microgrid.get_model_matrices()[1],
                                           charge_min_max=microgrid.charge_min_max,
                                           storage_power_min_max=microgrid.storage_power_min_max,
                                           grid_power_min_max=microgrid.grid_power_min_max)
    
    # Controller definition
    controller = Controller(optimisation_layer=optimisation_layer,
                            dataset=dataset,
                            hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers)
    if args.load_dir is not None:
        logging.info("Loading model from %s", args.load_dir)
        controller = utils.load_model(
            controller, utils.model_file(args.load_dir))

    # Create loss
    regression_loss = MSELoss()
    total_cost_loss = TotalCostLoss()

    # Create trainer
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=experiment_path, name="logs") if args.st_checkpoint_dir is None else None
    trainer = pl.Trainer(gpus=args.gpu, max_epochs=1, enable_checkpointing=False,
                         accelerator="auto", logger=tb_logger,
                         enable_progress_bar=True if args.st_checkpoint_dir is None else False)
    # Create Learner
    learner = OnlineLearner(plant=microgrid,
                            controller=controller,
                            regression_loss=regression_loss,
                            total_cost_loss=total_cost_loss,
                            dataset=dataset,

                            learning_rate=args.learning_rate,

                            weight_decay_start_weight=args.weight_decay_start_weight,
                            weight_decay_end_weight=args.weight_decay_start_weight,
                            weight_decay_start_time_step=args.weight_decay_start_time_step,
                            weight_decay_end_time_step=args.weight_decay_end_time_step,

                            swa_gamma=args.swa_gamma,
                            swa_start_time_step=args.swa_start_time_step,
                            swa_end_time_step=args.swa_end_time_step,
                            swa_replace_frequency=args.swa_replace_frequency,

                            mse_start_weight=args.mse_start_weight,
                            mse_exp_decay=args.mse_exp_decay,
                            mse_end_weight=args.mse_end_weight,
                            mse_start_time_step=args.mse_start_time_step,
                            mse_end_time_step=args.mse_end_time_step,

                            task_start_weight=args.task_start_weight,
                            task_end_weight=args.task_end_weight,
                            task_start_time_step=args.task_start_time_step,
                            task_end_time_step=args.task_end_time_step,
                            task_window=args.task_window)

    # Fit the model
    trainer.fit(learner, train_dataloaders=dataloader)

    # Plot results
    plots.plot_trajectory(experiment_path=experiment_path,
                          state=learner.state, constraints_state=microgrid.charge_min_max,
                          controller_outputs=learner.controller_outputs,
                          constraints_input=[microgrid.storage_power_min_max, microgrid.storage_power_min_max, []],
                          t_initial=0, t_final=learner.state.shape[0])

    # Create empty results dictionary
    results = {}
    if args.plot:
        data_splits = ['train', 'val', 'test']
    else:
        data_splits = ['test']

    for data_split in data_splits:
        results[data_split] = {}
        plots.plot_predictions(experiment_path=experiment_path,
                               dataset=dataset,
                               predictions=learner.predictions[data_split],
                               targets=learner.targets[data_split],
                               output_features=args.output_features,
                               t_initial=0, t_final=learner.predictions[data_split].shape[0],
                               data_split=data_split)

        plots.plot_decision_variables(experiment_path=experiment_path,
                                      decision_variables=learner.decisions[data_split],
                                      constraints={
                                          'Pl': [dataset.get_profile()[:, 1]],
                                          'Pr': [dataset.get_profile()[:, 0]],
                                          'Ps_in': microgrid.storage_power_min_max,
                                          'Ps_out': microgrid.storage_power_min_max,
                                          'Pg': microgrid.grid_power_min_max,
                                          'eps': [1.0, -1.0]},
                                      data_split=data_split)
        plots.plot_and_log_total_cost(experiment_path=experiment_path,
                                      price=dataset.get_profile()[:, 2],
                                      Pg=learner.decisions[data_split]["Pg"],
                                      results=results[data_split],
                                      data_split=data_split,
                                      prediction_horizon=args.prediction_horizon)
    # Store results
    results["Total cost [eur]"] = plots.total_cost(price=dataset.get_profile()[:, 2],
                                                   Pg=learner.controller_outputs[:, 2])
    utils.store_metrics(
        results, metrics=learner.metrics["train"], prefix="train")
    utils.store_metrics(results, metrics=learner.metrics["val"], prefix="val")
    utils.store_metrics(
        results, metrics=learner.metrics["test"], prefix="test")

    logging.info("Results: %s", results)

    # Save model
    utils.save_model(controller, utils.model_file(experiment_path))
    # Save results
    utils.save_pickle(results, utils.results_file(experiment_path))
    # Save learner
    utils.save_learner(learner, utils.learner_file(experiment_path))
    utils.save_learner_as_csv(learner, experiment_path)
    # Save dataset
    utils.save_dataset(dataset, utils.dataset_file(experiment_path))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument_group("Data")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='The directory where the data is stored.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='The number of workers to be used for loading the data.')
    parser.add_argument('--st_checkpoint_dir', type=str, default=None,
                        help='Directory where syne-tune checkpoints are stored.')

    parser.add_argument_group("Microgrid Data")
    parser = CSVDataset.add_specific_args(parser)

    parser.add_argument_group("Network")
    parser = Controller.add_specific_args(parser)

    parser.add_argument_group("Plant")
    parser = MicroGridSystem.add_specific_args(parser)

    parser.add_argument_group("Experiment")
    parser.add_argument('--seed', type=int, default=42,
                        help='The seed to be used for training.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu to be used for training.')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='The directory where the experiment results are stored.')
    parser.add_argument('--load_dir', type=str, default=None,
                        help='The directory where the model is loaded from. Default: None')
    parser.add_argument('--plot', type=int, choices=[0, 1], default=1,
                        help='Enable the plot of the train and validation results.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment directory.')

    parser = OnlineLearner.add_specific_args(parser)
    args = parser.parse_args()

    main(args)
