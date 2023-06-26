# Learning-based Model Predictive Control for Microgrid Energy Management

by Vittorio Casagrande (vittorio.casagrande.19@ucl.ac.uk), Martin Ferianc, Miguel Rodrigues and Francesca Boem

This repository contains the code used for the paper "An Online Learning Method for Microgrid Energy Management Control" presented at the 31st Mediterranean Conference on Control and Automation (MED 2023) in Limassol, Cyprus. 
In the code we implement an innovative method for online training of a neural network used for prediction of unknown profiles (for example load demand and electricity prices) to be used for microgrid energy management.
The code is implemented in Pytorch leveraging the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework for neural network training and [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) for constructing the convex optimisation layer.

- [Learning-based Model Predictive Control for Microgrid Energy Management](#learning-based-model-predictive-control-for-microgrid-energy-management)
  - [Abstract](#abstract)
  - [Running the code](#running-the-code)
    - [Requirements](#requirements)
    - [Running the experiment](#running-the-experiment)
  - [Citation](#citation)
  - [Authors](#authors)
  - [Contributing](#contributing)
  - [License](#license)

## Abstract
A novel Model Predictive Control (MPC) scheme based on online-learning (OL) for microgrid
energy management, is proposed.
The MPC method deals with uncertainty on the load demand, renewable generation and electricity prices, by employing the predictions provided by an online trained neural network in the optimisation problem. 
In order to adapt to possible changes in the environment, the neural network is online trained based on continuously received data.
The network hyperparameters can be selected by performing a hyperparameter optimisation before the deployment of the controller, using a pretraining dataset.
We show the effectiveness of the proposed method for microgrid energy management through extensive experiments on real microgrid datasets.
Moreover, we show that the proposed algorithm has good transfer learning (TL) capabilities among different microgrids.

## Running the code
Several experiments can be run using the code in this repository.
The power and price profiles for three different industrial sites are provided in the [data](data) folder.
A standard LSTM neural network is used for prediction of the unknown profiles, all the network and optimisation parameters can be passed as command line arguments.

### Getting the code
A copy of all the files can be obtained by cloning the repository:
```bash
git clone https://github.com/vittpi/ol-ems.git
```
### Requirements
A working Python environment is required to run the code.
To do this create a virtual environment and install the required packages.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the experiment
To run the experiments with default parameters, run the following command:
```bash
python3 main.py
```
Additional command line arguments can be passed to the script to change the default parameters. For example, to change the number of hidden units of the neural network to 24, run the following command:
```bash
python3 main.py --hidden_dim 24
```

### Hyperparameters optimisation
The syne-tune library is used for hyperparameters optimisation.
To run the optimisation, run the following command:
```bash
python3 tune.py --config_file ./configurations/sample.py
```
where the hyperparameters to optimise are specified in the [configurations/sample.py](configurations/sample.py) file.

## Citation
If you use this code, please cite the following paper:
```
@inproceedings{casagrande2023online,
  title={An Online Learning Method for Microgrid Energy Management Control},
  author={Casagrande, Vittorio and Ferianc, Martin and Rodrigues, Miguel and Boem, Francesca},
  booktitle={2023 Mediterranean Conference on Control and Automation (MED)},
  pages={***--***},
  year={2023},
  organization={IEEE}
}
```

## Authors
Vittorio Casagrande (vittorio.casagrande.19@ucl.ac.uk), Martin Ferianc (martin.ferianc.19@ucl.ac.uk)

## Contributing
In case you find any bugs or have any suggestions, please open an issue or a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
