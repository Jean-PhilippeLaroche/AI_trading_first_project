import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    Observations: stock prices + technical indicators + portfolio info
    Actions: discrete - 0=Hold, 1=Buy, 2=Sell
    Reward: change in portfolio value
    """

    metadata = {'render.modes': ['human']}  # Optional, for render(), if I want to visualize episodes


    def __init__(self, df, initial_balance=10000, window_size=30):
        """
        df: pandas DataFrame containing stock data with at least 'Close' column
        initial_balance: starting cash for the agent
        window_size: number of previous days to include in state
        """
        super(TradingEnv, self).__init__()

        # ----- Action space -----
        # 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # ----- Observation space -----
        # We'll use a vector of last `window_size` closing prices + cash + shares held
        # For simplicity, prices are normalized to start at 1
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(window_size + 2,),  # window_size prices + cash + shares held
            dtype=np.float32
        )

        # ----- Internal state -----
        self.df = df.reset_index(drop=True)  # Ensure indexing from 0
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size  # start after first window
        self.cash_balance = initial_balance
        self.num_shares = 0
        self.portfolio_value = initial_balance

        # History buffer for visualization/debugging
        self.history = {
            'step': [],
            'price': [],
            'cash': [],
            'shares': [],
            'portfolio_value': [],
            'action': [],
            'reward': []
        }


    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state for a new episode.
        Returns the initial observation.
        """
        super().reset(seed=seed)  # For reproducibility

        # ----- Reset internal variables -----
        self.current_step = self.window_size  # start after first window
        self.cash_balance = self.initial_balance
        self.num_shares = 0
        self.portfolio_value = self.initial_balance

        # ----- Clear history buffer -----
        self.history = {
            'step': [],
            'price': [],
            'cash': [],
            'shares': [],
            'portfolio_value': [],
            'action': [],
            'reward': []
        }

        # ----- Prepare initial observation -----
        prices_window = self.df['close'].iloc[self.current_step - self.window_size:self.current_step].values
        normalized_prices = prices_window.flatten() / prices_window[0]  # ensure 1D

        obs = np.concatenate([
            normalized_prices,
            np.array([self.cash_balance, self.num_shares], dtype=np.float32)
        ]).astype(np.float32)

        # ----- Optional info dictionary for debugging -----
        info = {
            'portfolio_value': self.portfolio_value,
            'current_price': float(self.df['close'].iloc[self.current_step - 1]),
            'cash_balance': self.cash_balance,
            'num_shares': self.num_shares
        }

        return obs, info

    def step(self, action):
        """
        Executes one time step in the environment.
        action: 0=Hold, 1=Buy, 2=Sell
        Returns: observation, reward, done, info
        """
        # ----- Get current price -----
        current_price = self.df['close'].iloc[self.current_step]

        # ----- Apply action -----
        if action == 1:  # Buy
            # Buy as many shares as possible with current cash
            shares_to_buy = int(self.cash_balance // current_price)
            self.cash_balance -= shares_to_buy * current_price
            self.num_shares += shares_to_buy

        elif action == 2:  # Sell
            # Sell all shares
            self.cash_balance += self.num_shares * current_price
            self.num_shares = 0


        # ----- Update portfolio value -----
        self.portfolio_value = self.cash_balance + self.num_shares * current_price

        # ----- Calculate reward -----
        previous_value = self.history['portfolio_value'][-1] if self.history[
            'portfolio_value'] else self.initial_balance
        reward = self.portfolio_value - previous_value

        # ----- Log history -----
        self.history['step'].append(self.current_step)
        self.history['price'].append(current_price)
        self.history['cash'].append(self.cash_balance)
        self.history['shares'].append(self.num_shares)
        self.history['portfolio_value'].append(self.portfolio_value)
        self.history['action'].append(action)
        self.history['reward'].append(reward)

        # ----- Move to next step -----
        self.current_step += 1
        done = self.current_step >= len(self.df)

        # ----- Prepare next observation -----
        if done:
            # Return a dummy observation at the end to avoid slicing errors
            obs = np.zeros(self.window_size + 2, dtype=np.float32)
        else:
            start_idx = max(self.current_step - self.window_size, 0)  # safe slicing
            prices_window = self.df['close'].iloc[start_idx:self.current_step].values

            # Flatten to 1D and normalize prices relative to first in window
            prices_flat = prices_window.flatten()
            normalized_prices = prices_flat / prices_flat[0]

            # Concatenate with cash balance and number of shares
            obs = np.concatenate((
                normalized_prices.flatten(),  # ensure 1D
                np.array([float(self.cash_balance), float(self.num_shares)], dtype=np.float32)  # scalars only
            )).astype(np.float32)

        # ----- Extra info for debugging -----
        info = {
            'portfolio_value': self.portfolio_value,
            'current_price': current_price,
            'cash_balance': self.cash_balance,
            'num_shares': self.num_shares
        }

        return obs, reward, done, info


    def render(self, mode='human'):
        """
        Render the environment for visualization.
        mode: 'human' for live print/plot, 'rgb_array' for image frames (optional)
        """
        if mode == 'human':
            # ----- Simple textual output -----
            print(f"Step: {self.current_step}")
            print(f"Price: {float(self.df['close'].iloc[self.current_step - 1]):.2f}")
            print(f"Cash: {float(self.cash_balance):.2f}, Shares: {self.num_shares}")
            print(f"Portfolio Value: {float(self.portfolio_value):.2f}")

if __name__ == "__main__":
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    experiment_counter = 0
    returns = []

    while experiment_counter < 400:

        # sample AAPl data
        ticker = 'AAPL'
        df = yf.download(ticker, start='2023-01-01', end='2023-06-01')

        # environment
        env = TradingEnv(df, window_size=10, initial_balance=10000)

        # random agent for 1 episode
        obs, info = env.reset()
        done = False

        while not done:
            action = np.random.choice([0, 1, 2])  # random: Hold, Buy, Sell
            obs, reward, done, info = env.step(action)
            env.render(mode='human')  # print current step info

        # ----- Episode finished -----
        print("\nEpisode finished!")
        print(f"Final Portfolio Value: {float(env.portfolio_value):.2f}")

        rate_of_return = ((env.portfolio_value * 100) / 10000) - 100
        print(f"Rate of return: {float(rate_of_return):.2f}%")

        returns.append(rate_of_return)
        experiment_counter += 1

    # ----- After all experiments: plot histogram -----
    returns = np.array(returns, dtype=float)

    # Define custom bin edges (2.5% steps)
    bin_width = 2.5
    min_return = returns.min()
    max_return = returns.max()
    bins = np.arange(np.floor(min_return / bin_width) * bin_width,
                         np.ceil(max_return / bin_width) * bin_width + bin_width,
                         bin_width)

    # Plot histogram
    plt.figure(figsize=(8, 5))
    counts, edges, patches = plt.hist(returns, bins=bins, edgecolor='black', alpha=0.7, density=True)

    # Add fitted normal distribution curve
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    x = np.linspace(min_return, max_return, 200)
    pdf = norm.pdf(x, mean_return, std_return)
    plt.plot(x, pdf, 'r-', linewidth=2, label="Normal fit")

    # Add mean line
    plt.axvline(mean_return, color='blue', linestyle='dashed', linewidth=1.5,
                    label=f"Mean = {mean_return:.2f}%")

    # Labels & legend
    plt.title("Distribution of Rates of Return (Random Trading Agent)")
    plt.xlabel("Rate of Return (%)")
    plt.ylabel("Density")
    plt.legend()

    plt.show()

    # ----- Optional: plot history -----
    # TODO: Add code to visualize price vs portfolio vs actions
    # Example: env._plot_history()