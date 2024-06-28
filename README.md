# Enhanced Quantitative Trading Strategies through Sentiment Analysis Using Large Language Models

This repository contains the implementation of a comprehensive information integration system for high-frequency trading (HFT) in stock markets. Our approach integrates nested decision execution framework for HFT operations and LLM for extracting and analyzing financial sentiments from news and social media.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [FinLLAMA](#finetuning-finllama)
- [SimonsHFT](#simonshft)
- [Evaluation](#evaluation-tbd)
  - [Running Backtests with Qlib](#running-backtests-with-qlib)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project leverages the Qlib framework for high-frequency trading operations and integrates advanced sentiment analysis using FinLLAMA. By incorporating real-time sentiment data from news and social media, we aim to enhance the decision-making process and profitability in volatile stock markets.

## Getting Started

### Installation

To set up the environment, you need to install the required dependencies for both Qlib and FinLLAMA. Ensure you have Python 3.8.x.

1. Clone the repository:

```bash
    git clone https://github.com/Flemington7/SimonsAgent.git
```

2. Install the required Python packages: (TBD)

(Will be updated with the required packages for both FinLLAMA and Qlib, now you can install the packages from the specific directories)

```bash
    pip install -r requirements.txt
```
> [!TIP]
>If you are using a Windows machine, you can install the version of the required packages using the following command:

  ```bash
      pip install -r requirements-win.txt
  ```

### Data Preparation

(Will be updated with the required data preparation steps for both FinLLAMA and Qlib)

1. **Sentiment Data**: Prepare the sentiment analysis data by processing news and social media texts using FinLLAMA. Ensure that the data is in a format compatible with the finetuning process.

2. **Financial Data**: Download and prepare the financial data as required by Qlib. Follow the Qlib documentation for setting up data. As mentioned above, we will integrate the sentiment data with the financial data, and convert it into a format that suitable for backtesting.

## [FinLLAMA](/src/FinLLAMA/README.md)

## [SimonsHFT](/src/SimonsHFT/README.md)

## Evaluation (TBD)

Evaluate the performance of the trading strategies and the impact of sentiment analysis on trading decisions. Compare the results with and without sentiment data integration and the benchmark performance.

### Running Backtests with Qlib

Set up and run backtests using Qlib with the integrated sentiment data:

```python
python SimonsHFT/workflow.py backtest
```

## Contributing

**W. Ye** designed the workflow, deployed and fine-tuned the language models, developed the trading strategies, performed data analysis and wrote the manuscript.  
**H. Li** conducted the related work review, forecast model selection, backtesting and wrote the poster.  
**J. Li** conducted interference of LLM and integrated the sentiment analysis with traditional price-volume features.  
All authors contributed to the design of the study.

### Acknowledgment

This work was supported by Jiachen Wang, Wentao Ye, Rui Chen and Hanyu Wei, who provided valuable computational resources.  
We also thank the Qlib project in Microsoft Research Asia for providing the open-source quantitative investment platform and the Llama project in Meta AI for providing the pre-trained language models.

## License

This project is licensed under the terms of the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
