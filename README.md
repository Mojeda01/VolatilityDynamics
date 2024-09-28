 # Simulated Stock Prices With Live Update

This script simulates stock prices over time using a GBM model with dynamically updating drift (mu) and volatility (sigma). The simulation inmates live stock price movements with Matplotlib, updating the x-axis and y-axis as the simulation progresses.

## Features

- **Dynamic Drift and Volatility:**
    - The drift(`mu`) and volatility(`sigma`) evolve over time, simulating real-world market conditions where both can change dynamically.

- **Real-Time Plot Update**
    - The plot of simulated stock prices updates in real-tme, extending the x-axis and adjusting the y-axis as new prices are generated.

## Components

### `fMu()`

Generates a time-evolving drift (`mu`) using a mean-reverting Ornstein-Uhlenbeck process, which is commonly used to model stock returns reverting to a long-term mean.

**Parameters:**

- `n_steps`: Number of time steps in the simulation.
- `mu_long_term`: The long-term average value for the drift.
- `mean_reversion_speed`: The speed at which the drift reverts to the long-term mean.
- `volatility_mu`: Controls the volatility of the drift.

### `fSigma()`

Generates a time-evolving volatility (sigma) using a random-walk process. The volatility fluctuates, and a lower bound is enforced to prevent it from becoming negative.

**Parameters**

- `n_steps`: Number of time steps in the simulation.
- `sigma_base`: The starting value of volatility.
- `volatility_vol`: The magnitude of fluctuations in volatility.

### `simulate_prices()`

Simulates stock prices over time using GBM. The function updates stock prices based on dynamic my and sigma values generated by `fMu()` and `fSigma()`.

**Parameters:**

- `n`: Number of simulation steps per call.
- `k`: Number of stock price realizations.
- `mu`: The drift values for each step.
- `sigma`: The volatility values for each step.
- `delta_t`: The time step size.

### `update()`

Updates the plot with new stock prices for each frame in the animation.

## Usage

### Prerequisites

- Python 3.x
- Matplotlib for plotting and animation
- NumPy for numerical computations.

## Run the script.

1. Run the script: `python3 exec.py`

The simulation will start, and you will see the stock prices being plotted live. The drift and volatility of the stock prices will change dynamically over time, creating a more realistic market simulation.

## Customization

You can customize the parameters for drift and volatility `fMu()` and `fSigma()` functions to match different market conditions. For example:
    - Adjust `mu_long_term` for more bullish or bearish trends.
    - Changes `volatility_mu` or `volatility_vol` to simulate periods of high or low volatility.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

