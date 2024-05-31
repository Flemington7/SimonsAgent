# COMPREHENSIVE Information Integration for High-Frequency Trading using Qlib and FinLLAMA

This repository contains the implementation of a comprehensive information integration system for high-frequency trading (HFT) in stock markets. Our approach integrates nested decision execution framework for HFT operations and LLM for extracting and analyzing financial sentiments from news and social media.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [FinLLAMA](#finetuning-finllama)
- [SimonsHFT](#simonshft)
- [Backtesting with Qlib](#backtesting-with-qlib)
  - [Setting Up Qlib](#setting-up-qlib)
  - [Running Backtests](#running-backtests)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project leverages the Qlib framework for high-frequency trading operations and integrates advanced sentiment analysis using FinLLAMA. By incorporating real-time sentiment data from news and social media, we aim to enhance the decision-making process and profitability in volatile cryptocurrency markets.

## Getting Started

### Installation

To set up the environment, you need to install the required dependencies for both Qlib and FinLLAMA. Ensure you have Python 3.8.x.

1. Clone the repository:

    ```bash
    git clone https://github.com/Flemington7/SimonsAgent.git
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1. **Sentiment Data**: Prepare the sentiment analysis data by processing news and social media texts using FinLLAMA. Ensure that the data is in a format compatible with the finetuning process.

2. **Financial Data**: Download and prepare the financial data as required by Qlib. Follow the Qlib documentation for setting up data. As mentioned above, we will integrate the sentiment data with the financial data, and convert it into a format that suitable for backtesting.

## [FinLLAMA](FinLLAMA/README.md)

## [SimonsHFT](SimonsHFT/README.md)

## Backtesting with Qlib

### Running Backtests

Set up and run backtests using Qlib with the integrated sentiment data:

```python
python SimonsHFT/workflow.py backtest
```

## Evaluation (TBD)

Evaluate the performance of the trading strategies and the impact of sentiment analysis on trading decisions. Compare the results with and without sentiment data integration and the benchmark performance.

## Contributing (TBD)

## License

This project is licensed under the terms of the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
