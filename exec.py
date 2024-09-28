# MIT License
# 
# Copyright (c) 2024 Marco Ojeda
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def fMu(n_steps = 1000, mu_long_term=2, mean_reversion_speed=0.1, volatility_mu=0.02):
    mu = np.zeros(n_steps)
    mu[0] = mu_long_term # Start drift at long-term
    for t in range(1, n_steps):
        # Ornstein-Uhnlenbeck process for mean_reverting drift
        mu[t] = mu[t-1] + mean_reversion_speed * (mu_long_term - mu[t-1] + volatility_mu * 
                                                  np.random.normal())
    return mu

def fSigma(n_steps=1000, sigma_base=0.2, volatility_vol=0.05):
    sigma = np.zeros(n_steps)
    sigma[0] = sigma_base # Start volatility at a base level
    for t in range(1, n_steps):
        # Random walk with volatility
        sigma[t] = sigma[t-1] + volatility_vol * np.random.normal()
        # Ensure volatility stays positive
        sigma[t] = max(sigma[t], 0.01) # Prevent sigma from becoming negative
    return sigma


def simPModel():
    # Time parameters
    T = 2.0  # Simulate over a longer time period (e.g., 2 years)
    delta_t = 1.0 / 12.0  # Use a larger time step (e.g., monthly instead of daily)
    n = int(T / delta_t)  # Total number of samples

    S0 = 100  # Higher initial price of stock to see clearer movements
    mu = fMu() # Higher drift
    sigma = fSigma()  # Higher volatility
    k = 5  # Reduce the number of realizations to avoid clutter

    # Store cumulative data for dynamic updates
    data = {'x_data': [], 'p_data': []}  # Use a dictionary to store mutable data

    def simulate_prices(n, k, mu, sigma, delta_t):
       r = np.zeros((n, k)) # Initialize an array to hold returns
       for i in range(n):
           # Indexing mu and sigma to take their values at the curret step (i)
           current_mu = mu[i]
           current_sigma = sigma[i]
           # Apply the Geometric Brownian Motion formula
           r[i] = (current_mu - 0.5 * current_sigma ** 2) * delta_t + current_sigma * np.sqrt(delta_t) * np.random.normal(size=k)
           # Cumulative sum for stock prices (note that we take exp to get the GBM result)
           p = 100 * np.exp(np.cumsum(r, axis=0))
           return p

    fig, ax = plt.subplots()
    lines = [ax.plot([], [])[0] for _ in range(k)]  # Fewer lines for fewer realizations
    ax.set_ylim(0, 200)  # Adjust y-axis limits based on expected stock prices

    # Initialize the plot
    def init():
        ax.set_xlim(0, 10)  # Start with a smaller x-limit and extend it dynamically
        for line in lines:
            line.set_data([], [])
        return lines

    # Update the plot for each frame
    def update(frame):
        p = simulate_prices(1, k, mu, sigma, delta_t)  # Simulate only one step per frame

        if len(data['x_data']) == 0:
            data['x_data'].append(0)  # Initial x-data
            data['p_data'].append(p[0, :])  # Set initial prices for 3 lines
        else:
            data['x_data'].append(data['x_data'][-1] + 1)  # Increment x by 1 for each new frame
            data['p_data'].append(p[0, :])  # Append the new step prices for 3 lines

        # Convert p_data list to a numpy array for efficient indexing
        p_data_np = np.array(data['p_data'])

        ax.set_xlim(0, len(data['x_data']))  # Dynamically adjust the x-axis limits

        for i, line in enumerate(lines):
            line.set_data(data['x_data'], p_data_np[:, i])  # Update each line with new data
        return lines

    # Animate the plot without using `blit=True`
    ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, interval=100)
    plt.title(f'Simulated Stock Prices (Live Update)')
    plt.show()

simPModel()

