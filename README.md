
# 004 Pairs Trading Project

Project 004 for Microstructure and Trading systems class. This repository contains the implementation of a **Pairs Trading strategy**, including data collection, cointegration testing, signal generation, backtesting, and results visualization. The project is written in Python and designed for reproducibility and modularity.

---

## **Table of Contents**

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Setup Instructions](#setup-instructions)

   * [Clone the Repository](#1-clone-the-repository)
   * [Create Virtual Environment](#2-create-and-activate-a-virtual-environment-recommended)
   * [Install Dependencies](#3-install-dependencies)
* [Data Requirements](#data-requirements)
* [How to Run the Project](#how-to-run-the-project)
* [Outputs and Results](#output-and-results)
* [Requirements](#requirements)
* [License](#license)

---

## Project Overview

This project includes:

* Data download from Yahoo Finance
* Statistical tests (correlation and cointegration tests)
* Signal generation
* Backtesting engine
* Performance metrics and visualization

---

## Repository Structure

```
.
├─ backtest.py                 # Run historical simulations / strategy backtest  
├─ cointegration.py            # Cointegration testing functions  
├─ cointegration_results.xlsx  # Results of cointegration tests (output)  
├─ kalman.py                   # Kalman filter for dynamic hedge ratio estimation  
├─ main.py                     # Pipeline entry-point
├─ metrics.py                  # Metrics & evaluation utilities (returns, sharpe, drawdowns, etc.)  
├─ models.py                   # Model wrapper helpers if you decide to extend  
├─ plots.py                    # Plotting utilities  
├─ trials.py                   # Script for experiment / parameter trials  
├─ utils.py                    # Varied functions
├─ requirements.txt            # Python dependencies  
└─ README.md                   # This file  
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AnaSofiaHinojosa/004PairsTrading.git
cd 004PairsTrading
```

### 2. Create and activate a virtual environment (recommended)

#### **MacOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### **Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Requirements

The strategy requires **historical price data** for a pair of assets (e.g., stocks, ETFs).

### Data come from:

* Yahoo Finance via `yfinance`

### Minimum fields needed:

| Column  | Description                 |
| ------- | --------------------------- |
| `Date`  | Timestamp                   |
| `Close` | Closing price of each asset |

---

## How to Run the Project

### Run the full workflow

```bash
python main.py
```

This will:

1. Download historical data
2. Test cointegration
3. Build the spread & compute z-scores
4. Generate buy/sell signals
5. Run the backtest
6. Save plots & metrics

---

## Output and Results

Running the project will produce the following outputs:

**Terminal Output**
- Sector-by-sector cointegration testing logs  
- Cointegration summary table (top cointegrated pairs with eigenvectors and statistics)  
- Selected trading pair  
- Performance metrics (Sharpe, Sortino, Max Drawdown, Calmar)  
- Trade statistics (number of trades, wins and losses, win rate, avg win/loss, profit factor, commissions, borrow costs)  
- Final portfolio value and final cash balance  

**Generated Plots**
- Cointegrated pair prices
- Spread plot for the selected pair 
- Portfolio equity curve
- VECM normalized plot  
- VECM normalized plot with highlighted threshold regions
- Estimated vs. real P2 values  
- Estimated vs. real VECM values  
- Estimated vs. real e1/e2 values  
- Kalman filter weights over time  
- Distribution of returns per trade  

---

## Requirements

* Python 3.9+
* macOS / Linux / Windows
* Internet connection (for Yahoo Finance)

---

## License

This project is released under the **MIT License**.
See the **LICENSE** file for details.

---
