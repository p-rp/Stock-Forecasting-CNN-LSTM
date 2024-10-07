<!--
**       .@@@@@@@*  ,@@@@@@@@     @@@     .@@@@@@@    @@@,    @@@% (@@@@@@@@
**       .@@    @@@ ,@@          @@#@@    .@@    @@@  @@@@   @@@@% (@@
**       .@@@@@@@/  ,@@@@@@@    @@@ #@@   .@@     @@  @@ @@ @@/@@% (@@@@@@@
**       .@@    @@% ,@@        @@@@@@@@@  .@@    @@@  @@  @@@@ @@% (@@
**       .@@    #@@ ,@@@@@@@@ @@@     @@@ .@@@@@@.    @@  .@@  @@% (@@@@@@@@
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">COMP 432: Machine Learning Fall 2022</h3>

  <p align="center">
    Stock forecasting with CNN and LSTM
    <br />
    <a href="https://www.overleaf.com/read/pgyqxxjwqrhs"><strong>Link to report »</strong></a>
    <br />
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#welcome">Welcome</a></li>
    <li><a href="#team-members">Team Members</a></li>
    <li><a href="#installing-dependencies">Installing Dependencies</a></li>
    <li><a href="#data-collection">Data Collection</a></li>
    <li><a href="#Training">Training</a></li>
    <li><a href="#Cross-validation-and-testing">Cross-Validation and Testing</a></li>
  </ol>
</details>

<!-- Welcome -->

## Welcome

Welcome to our COMP 432 project. Our objective was to create a stock predictor using machine learning strategies; in this case using neural nets in PyTorch. Please refer to our report for more details, .py files to see under the hood, and our notebooks to observe the model in action.

<!-- TEAM MEMBERS -->

## Team Members

<table>
<thead>
  <tr>
    <th>Name</th>
    <th>ID</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Frédéric Pelletier</td>
    <td>40173212</td>
  </tr>
  <tr>
    <td>Rami Rafeh</td>
    <td>29198024</td>
  </tr>
  <tr>
    <td>Vaansh Vikas Lakhwara</td>
    <td>40114764</td>
  </tr>
   <tr>
    <td>Piyush Pokharkar</td>
    <td>40120654</td>
  </tr>
</tbody>
</table>

## Installing Dependencies

- python -m pip install -r requirements.txt (if you use pip and venv)
- conda install --file requirements.txt (if you use conda)

## Data Collection

Usage:

```sh
$ python data_preprocessing/main.py --help
usage: main.py [-h] [-f] [-d D]

Data collection and preprocessing

options:
  -h, --help   show this help message and exit
  -f, --fresh  generate a new csv file with all stock symbols
  -d D         the directory to store results in
```

In the current directory run:

```sh
sh scripts/run.sh
```

To get rid of all csv files in `data`:

```sh
sh scripts/clean.sh
```

## Training

- training/model/
  - [module](https://github.com/fredpell1/comp432-project/blob/main/training/model/net.py) containing the model implementation and a [notebook](https://github.com/fredpell1/comp432-project/blob/main/training/model/modelexample.ipynb) showcasing how to train the model on a dummy example
- training/
  - [module](https://github.com/fredpell1/comp432-project/blob/main/training/plot_utils.py) containing utility functions to plot the model's predictions
  - [module](https://github.com/fredpell1/comp432-project/blob/main/training/tensor_utils.py) containing utility functions to manipulate torch tensors
  - [module](https://github.com/fredpell1/comp432-project/blob/main/training/training.py) containing utility functions to train the model with different set of hyperparameters

## Cross-Validation and Testing

- [module](https://github.com/fredpell1/comp432-project/blob/main/testing.py) containing utility functions for cross-validation and testing the model
- [notebook](https://github.com/fredpell1/comp432-project/blob/main/cross_validation.ipynb) to perform cross-validation. It will save the best set of hyperparameters [here](https://github.com/fredpell1/comp432-project/blob/main/output/hyperparameters.json)
- [notebook](https://github.com/fredpell1/comp432-project/blob/main/training.ipynb) to perform training and testing of the model, using the hyperparameters found by cross-validation. It will save accuracy metrics [here](https://github.com/fredpell1/comp432-project/blob/main/output/metrics.json)
