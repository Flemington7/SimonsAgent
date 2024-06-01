# SimonsHFT: Nested Decision Execution Framework for High-Frequency Trading

## Overview

The `SimonsHFT` is a nested decision execution workflow designed to support high-frequency trading (HFT) by integrating multiple levels of trading strategies. This framework allows for joint backtesting of daily and intraday trading strategies, considering their interactions to optimize overall performance.

## Table of Contents

- [Overview](#overview)
- [Introduction](#introduction)
- [Framework Design](#framework-design)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Running the Workflow](#running-the-workflow)
  - [Information Extractor](#information-extractor-tbd)
  - [Training the Model](#training-the-model-tbd)
    - [Custom Model Training](#custom-model-training-tbd)
    - [Run example model](#run-example-model)
  - [Decision Generator](#decision-generator-tbd)
  - [Backtesting](#backtesting)
    - [Weekly Portfolio Generation and Daily Order Execution](#weekly-portfolio-generation-and-daily-order-execution)
    - [Daily Portfolio Generation and Minutely Order Execution](#daily-portfolio-generation-and-minutely-order-execution)


## Introduction

Daily trading (e.g., portfolio management) and intraday trading (e.g., order execution) are typically studied separately in quantitative investment. To achieve joint trading performance, these strategies must interact and be backtested together. The `NestedDecisionExecutionWorkflow` framework provided by `Qlib` supports multi-level joint backtesting strategies, allowing for accurate evaluation and optimization. Our `SimonsHFT` framework is built on top of this framework to support high-frequency trading operations.

## Framework Design

The `SimonsHFT` considers the interaction of strategies at multiple levels. Each level consists of a "Trading Agent" and "Execution Environment." The "Trading Agent" includes:
- Data processing module ("Information Extractor")
- Forecasting module ("Forecast Model")
- Decision generator ("Decision Generator")

The trading algorithm generates decisions based on forecast signals, which are then executed by the "Execution Environment", and returns the execution results.

## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/comprehensive-hft.git
    cd SimonsHFT
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1. **Financial Data**: Download and prepare the financial data as required by Qlib. Follow the Qlib documentation for setting up data. As mentioned before, we will integrate the sentiment data with the financial data, and convert it into a format that suitable for backtesting. 

## Running the Workflow

![Framework](framework.svg)

### Information Extractor (TBD)

As mentioned above, the `SimonsHFT` framework integrates advanced sentiment analysis using FinLLAMA. By incorporating real-time sentiment data from news and social media, we aim to enhance the decision-making process and profitability in volatile stock markets. We plan to use the FinLLAMA model to extract and analyze financial sentiments from news and social media and convert these Textual data into numerical data for further analysis.

(It will be one of the most important parts of the `SimonsHFT` framework!)

### Training the Model (TBD)

An increasing number of SOTA Quant research works/papers, which focus on building forecasting models to mine valuable signals/patterns in complex financial data, are released in `Qlib`. All these models are runnable with Qlib. Users can find the config files they provide and some details about the model through the [benchmarks](examples/benchmarks) folder. More information can be retrieved at the model files listed above.

### Custom Model Training (TBD)
(It will be one of the most important parts of the `SimonsHFT` framework! Please refer to this [link](#https://qlib.readthedocs.io/en/latest/start/integration.html) for more details!)

#### Run example model

`Qlib` provides three different ways to run a single model, users can pick the one that fits their cases best:

- Users can use the tool `qrun` mentioned above to run a model's workflow based from a config file.

- Users can create a `workflow_by_code` python script based on the [one](examples/workflow_by_code.py) listed in the `examples` folder.

- Users can use the script [`run_all_model.py`](examples/run_all_model.py) listed in the `examples` folder to run a model. Here is an example of the specific shell command to be used: `python run_all_model.py run --models=lightgbm`, where the `--models` arguments can take any number of models listed above(the available models can be found  in [benchmarks](examples/benchmarks/)). For more use cases, please refer to the file's [docstrings](examples/run_all_model.py).
    - **NOTE**: Each baseline has different environment dependencies, please make sure that your python version aligns with the requirements(e.g. TFT only supports Python 3.6~3.7 due to the limitation of `tensorflow==1.15.0`)

`Qlib` also provides a script [`run_all_model.py`](examples/run_all_model.py) which can run multiple models for several iterations.
 
- **NOTE**: the script only support *Linux* for now. Other OS will be supported in the future. Besides, it doesn't support parallel running the same model for multiple times as well.

The script will create a unique virtual environment for each model, and delete the environments after training. Thus, only experiment results such as `IC` and `backtest` results will be generated and stored.

Here is an example of running all the models for 10 iterations:

```python
python run_all_model.py run 10
```

It also provides the API to run specific models at once. For more use cases, please refer to the file's [docstrings](examples/run_all_model.py). 

### Decision Generator (TBD)

In `examples/benchmarks` we have various **alpha** models that predict the stock returns. We also use a simple rule based `TopkDropoutStrategy` to evaluate the investing performance of these models. However, such a strategy is too simple to control the portfolio risk like correlation and volatility. To this end, an optimization based strategy should be used to for the trade-off between return and risk. Please refer to the [Portfolio Optimization Strategy](examples/portfolio/README.md) for more details.

### Backtesting

This [workflow](#workflow.py) is an example for nested decision execution in backtesting. Qlib supports nested decision execution in backtesting. It means that users can use different strategies to make trade decision in different frequencies.

#### Weekly Portfolio Generation and Daily Order Execution

This [workflow](#workflow.py) provides an example that uses a DropoutTopkStrategy (a strategy based on the daily frequency Lightgbm model) in weekly frequency for portfolio generation and uses SBBStrategyEMA (a rule-based strategy that uses EMA for decision-making) to execute orders in daily frequency. 

Start backtesting by running the following command:

```bash
python workflow.py backtest
```

After running the [workflow](#workflow.py), you will see the following output like this (at this time, we run the weekly portfolio generation and minutely order execution):

```bash
[21830:MainThread](2024-05-30 21:35:06,100) INFO - qlib.workflow - [record_temp.py:515] - Portfolio analysis record 'port_analysis_1day.pkl' has been saved as the artifact of the Experiment 1
'The following are analysis results of benchmark return(1day).'
                       risk
mean              -0.000225
std                0.001969
annualized_return -0.053616
information_ratio -1.764751
max_drawdown      -0.027478
'The following are analysis results of the excess return without cost(1day).'
                       risk
mean               0.001371
std                0.015283
annualized_return  0.326362
information_ratio  1.384220
max_drawdown      -0.213554
'The following are analysis results of the excess return with cost(1day).'
                       risk
mean               0.001363
std                0.015282
annualized_return  0.324297
information_ratio  1.375560
max_drawdown      -0.213554
[21830:MainThread](2024-05-30 21:35:06,123) INFO - qlib.workflow - [record_temp.py:540] - Indicator analysis record 'indicator_analysis_1day.pkl' has been saved as the artifact of the Experiment 1
'The following are analysis results of indicators(1day).'
        value
ffr  0.960916
pa   0.000476
pos  0.562500
[21830:MainThread](2024-05-30 21:35:06,708) INFO - qlib.timer - [log.py:127] - Time cost: 0.000s | waiting `async_log` Done
```

#### Daily Portfolio Generation and Minutely Order Execution

This [workflow](#workflow.py) also provides a high-frequency example that uses a DropoutTopkStrategy for portfolio generation in daily frequency and uses SBBStrategyEMA to execute orders in minutely frequency. 

Start backtesting by running the following command:
```bash
python workflow.py backtest_highfreq
```


