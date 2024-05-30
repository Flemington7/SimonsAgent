# Nested Decision Execution Framework for High-Frequency Trading

## Overview

The `NestedDecisionExecutionWorkflow` is designed to support high-frequency trading (HFT) by integrating multiple levels of trading strategies. This framework allows for joint backtesting of daily and intraday trading strategies, considering their interactions to optimize overall performance.

## Table of Contents

- [Overview](#overview)
- [Introduction](#introduction)
- [Framework Design](#framework-design)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Running the Workflow](#running-the-workflow)
  - [Training the Model](#training-the-model)
  - [Backtesting](#backtesting)
  - [Collecting Data](#collecting-data)
  - [Checking Frequency Differences](#checking-frequency-differences)
  - [Daily Backtest](#daily-backtest)
- [License](#license)

## Introduction

Daily trading (e.g., portfolio management) and intraday trading (e.g., order execution) are typically studied separately in quantitative investment. To achieve joint trading performance, these strategies must interact and be backtested together. The `NestedDecisionExecutionWorkflow` framework supports multi-level joint backtesting strategies, allowing for accurate evaluation and optimization.

## Framework Design

The nested decision execution framework considers the interaction of strategies at different levels. Each level consists of a "Trading Agent" and "Execution Environment." The "Trading Agent" includes:
- Data processing module ("Information Extractor")
- Forecasting module ("Forecast Model")
- Decision generator ("Decision Generator")

The trading algorithm generates decisions based on forecast signals, which are then executed by the "Execution Environment." This nested structure allows for the evaluation and optimization of trading strategies at multiple levels.

## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/comprehensive-hft.git
    cd comprehensive-hft
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1. **Financial Data**: Download and prepare the financial data as required by Qlib. Follow the Qlib documentation for setting up your data.

## Running the Workflow


### Output Example

After running the workflow, you will see the following output:

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


## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.