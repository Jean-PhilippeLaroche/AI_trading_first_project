"""
backtest.py
-----------
Proper backtesting framework for stock trading strategies.
Simulates step-by-step trading with realistic constraints:
- No lookahead bias (predict one day at a time)
- Transaction costs (2% for commission + bid-ask spread)
- Position sizing constraints
- Comprehensive performance metrics

Key metrics calculated:
- Total return
- Expected return (average daily return)
- Sharpe ratio (risk-adjusted return)
- Max drawdown (worst peak-to-trough loss)
- Win rate (% of profitable trades)
"""

import numpy as np
import pandas as pd
import logging
import torch
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)


class Backtester:
    """
    Backtesting engine for trading strategies based on price predictions.
    """

    def __init__(self,
                 model,
                 scaler,
                 df,
                 feature_columns,
                 window_size=60,
                 initial_balance=10000,
                 transaction_cost_pct=0.02,  # 2% total cost
                 position_size_pct=0.95,  # Use 95% of available cash
                 max_shares_per_trade=100,
                 threshold=0.01):  # 1% threshold for signals
        """
        Initialize backtesting engine.

        Args:
            model: Trained PyTorch model
            scaler: Fitted MinMaxScaler from training
            df: DataFrame with price data and indicators
            feature_columns: List of feature column names (must match scaler)
            window_size: Number of days for prediction input
            initial_balance: Starting cash
            transaction_cost_pct: Total transaction cost (commission + spread)
            position_size_pct: Fraction of cash to use per trade
            max_shares_per_trade: Maximum shares to buy in single trade
            threshold: Relative price change threshold for buy/sell signals
        """
        self.model = model
        self.scaler = scaler
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.position_size_pct = position_size_pct
        self.max_shares_per_trade = max_shares_per_trade
        self.threshold = threshold

        # State variables
        self.cash = initial_balance
        self.shares = 0
        self.portfolio_history = []
        self.trades = []

        # Set model to evaluation mode
        self.model.eval()

        # Detect which device the model is on
        self.device = next(self.model.parameters()).device
        logging.info(f"Backtester initialized with model on device: {self.device}")

    def prepare_sequence(self, end_idx):
        """
        Prepare a single sequence for prediction (no lookahead).

        Args:
            end_idx: End index in dataframe (current timestep)

        Returns:
            torch.Tensor: Sequence ready for model prediction
        """
        start_idx = end_idx - self.window_size

        if start_idx < 0:
            raise ValueError(f"Not enough data: need {self.window_size} days, got {end_idx}")

        # Extract window of data
        window_df = self.df.iloc[start_idx:end_idx][self.feature_columns]

        # Scale the data (using pre-fitted scaler)
        window_scaled = self.scaler.transform(window_df)

        # Convert to tensor with batch dimension: (1, window_size, features)
        sequence = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        return sequence

    def predict_next_price(self, end_idx):
        """
        Predict the next day's price based on historical data up to end_idx.

        Args:
            end_idx: Current timestep in dataframe

        Returns:
            float: Predicted price (unscaled, in original units)
        """
        sequence = self.prepare_sequence(end_idx)

        with torch.no_grad():
            prediction_scaled = self.model(sequence).item()

        # Inverse transform to get actual price
        # Create dummy array with all features
        dummy = np.zeros((1, len(self.feature_columns)))

        # Find index of target column (assuming first column is 'close')
        target_idx = 0  # Adjust if 'close' is not first in feature_columns
        if 'close' in self.feature_columns:
            target_idx = self.feature_columns.index('close')

        dummy[0, target_idx] = prediction_scaled
        dummy_df = pd.DataFrame(dummy, columns = self.feature_columns)

        # Inverse transform
        unscaled = self.scaler.inverse_transform(dummy_df)
        predicted_price = unscaled[0, target_idx]

        return predicted_price

    def predict_all_prices_batch(self, start_idx, end_idx, batch_size=1024):
        """
        Predict prices for all timesteps in range using efficient batching.

        Args:
            start_idx: Start index in dataframe
            end_idx: End index in dataframe
            batch_size: Number of sequences to process at once (default: 1024)
                       Increase to 2048-4096 for faster speed (still <2GB GPU memory)

        Returns:
            np.array: Predicted prices for each timestep (unscaled, in original units)
        """
        all_predictions_scaled = []

        # Step 1: Prepare all sequences (fast on CPU)
        num_timesteps = end_idx - start_idx
        logging.info(f"Preparing {num_timesteps} sequences for batch prediction...")

        sequences = []
        failed_indices = []

        for i in range(start_idx, end_idx):
            try:
                seq = self.prepare_sequence(i)
                sequences.append(seq)
            except Exception as e:
                logging.warning(f"Failed to prepare sequence at index {i}: {e}")
                sequences.append(None)
                failed_indices.append(i - start_idx)

        # Step 2: Process in batches on GPU
        logging.info(f"Running batch predictions (batch_size={batch_size})...")
        num_batches = (len(sequences) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]

            # Separate valid and invalid sequences
            valid_indices = []
            valid_sequences = []
            for local_idx, seq in enumerate(batch_sequences):
                if seq is not None:
                    valid_indices.append(local_idx)
                    valid_sequences.append(seq)

            # Handle case where entire batch failed
            if len(valid_sequences) == 0:
                # All sequences in this batch failed - use fallback (current price or 0)
                all_predictions_scaled.extend([0.0] * len(batch_sequences))
                continue

            # Stack valid sequences into single batch tensor
            batch_tensor = torch.cat(valid_sequences, dim=0)  # Shape: (N, window, features)

            # Predict entire batch at once (this is the fast part!)
            with torch.no_grad():
                batch_predictions = self.model(batch_tensor)  # Shape: (N,)

            # Convert to numpy
            batch_predictions_np = batch_predictions.cpu().numpy()

            # Insert predictions back into correct positions
            batch_results = []
            pred_idx = 0
            for local_idx in range(len(batch_sequences)):
                if local_idx in valid_indices:
                    batch_results.append(float(batch_predictions_np[pred_idx]))
                    pred_idx += 1
                else:
                    # Failed sequence - use 0 as placeholder (will use fallback later)
                    batch_results.append(0.0)

            all_predictions_scaled.extend(batch_results)

            # Progress logging (every 10 batches or at end)
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                progress = ((batch_idx + 1) / num_batches) * 100
                logging.info(f"  Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%)")

        # Step 3: Inverse transform all predictions at once (vectorized - very fast!)
        logging.info("Inverse scaling predictions to original price units...")
        predictions_scaled = np.array(all_predictions_scaled)

        # Create dummy array with all features
        dummy = np.zeros((len(predictions_scaled), len(self.feature_columns)))

        # Find target column index
        target_idx = 0
        if 'close' in self.feature_columns:
            target_idx = self.feature_columns.index('close')

        # Fill target column with scaled predictions
        dummy[:, target_idx] = predictions_scaled

        # Convert to DataFrame for sklearn compatibility
        dummy_df = pd.DataFrame(dummy, columns=self.feature_columns)

        # Inverse transform all at once (much faster than loop)
        unscaled = self.scaler.inverse_transform(dummy_df)
        predictions_unscaled = unscaled[:, target_idx]

        # Handle failed sequences by using current price as fallback
        if failed_indices:
            logging.warning(f"Using fallback prices for {len(failed_indices)} failed predictions")
            for failed_idx in failed_indices:
                actual_idx = start_idx + failed_idx
                fallback_price = self.df.iloc[actual_idx]['close']
                predictions_unscaled[failed_idx] = fallback_price

        logging.info(f"Batch prediction complete: {len(predictions_unscaled)} prices predicted")

        return predictions_unscaled

    def generate_signal(self, predicted_price, current_price):
        """
        Generate trading signal based on prediction vs current price.

        Args:
            predicted_price: Model's price prediction
            current_price: Current market price

        Returns:
            str: 'BUY', 'SELL', or 'HOLD'
        """
        # Buy if predicted price is significantly higher
        if predicted_price > current_price * (1.0 + self.threshold):
            return 'BUY'

        # Sell if predicted price is significantly lower
        elif predicted_price < current_price * (1.0 - self.threshold):
            return 'SELL'

        else:
            return 'HOLD'

    def execute_trade(self, action, current_price, current_date):
        """
        Execute a trade with realistic constraints and costs.

        Args:
            action: 'BUY' or 'SELL'
            current_price: Current market price
            current_date: Current date (for logging)
        """
        if action == 'BUY' and self.cash > 0:
            # Calculate how many shares we can afford
            max_affordable = int((self.cash * self.position_size_pct) / current_price)
            shares_to_buy = min(max_affordable, self.max_shares_per_trade)

            if shares_to_buy > 0:
                # Calculate total cost including transaction fees
                gross_cost = shares_to_buy * current_price
                transaction_fee = gross_cost * self.transaction_cost_pct
                total_cost = gross_cost + transaction_fee

                # Execute if we have enough cash
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.shares += shares_to_buy

                    self.trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'gross_cost': gross_cost,
                        'transaction_fee': transaction_fee,
                        'total_cost': total_cost
                    })

                    logging.debug(f"BUY: {shares_to_buy} shares @ ${current_price:.2f} | Fee: ${transaction_fee:.2f}")

        elif action == 'SELL' and self.shares > 0:
            # Sell all shares
            shares_to_sell = self.shares
            gross_proceeds = shares_to_sell * current_price
            transaction_fee = gross_proceeds * self.transaction_cost_pct
            net_proceeds = gross_proceeds - transaction_fee

            self.cash += net_proceeds

            self.trades.append({
                'date': current_date,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'gross_proceeds': gross_proceeds,
                'transaction_fee': transaction_fee,
                'net_proceeds': net_proceeds
            })

            self.shares = 0

            logging.debug(f"SELL: {shares_to_sell} shares @ ${current_price:.2f} | Fee: ${transaction_fee:.2f}")

    def run(self, start_idx=None, end_idx=None):
        """
        Run the backtest step-by-step through the data.


        Args:
            start_idx: Start index (default: window_size)
            end_idx: End index (default: len(df))

        Returns:
            dict: Backtest results with portfolio history, trades, and metrics
        """
        if start_idx is None:
            start_idx = self.window_size
        if end_idx is None:
            end_idx = len(self.df)

        logging.info(f"Starting backtest from index {start_idx} to {end_idx}")
        logging.info(f"Initial balance: ${self.initial_balance:,.2f}")
        logging.info(f"Transaction cost: {self.transaction_cost_pct * 100}%")
        logging.info(f"Signal threshold: {self.threshold * 100}%")

        # Reset state
        self.cash = self.initial_balance
        self.shares = 0
        self.portfolio_history = []
        self.trades = []

        # ===================================================================
        # OPTIMIZATION: Batch predict all prices at once (1000x faster!)
        # ===================================================================
        all_predicted_prices = self.predict_all_prices_batch(
            start_idx,
            end_idx,
            batch_size=1024  # Adjust: 512 (slower, less memory) to 4096 (faster, more memory)
        )

        # Now loop through and process signals/trades (this is fast)
        logging.info("Processing trading signals and executing trades...")
        num_timesteps = end_idx - start_idx

        for idx, i in enumerate(range(start_idx, end_idx)):
            current_date = self.df.index[i]
            current_price = self.df.iloc[i]['close']
            predicted_price = all_predicted_prices[idx]

            # Generate trading signal
            signal = self.generate_signal(predicted_price, current_price)

            # Execute trade if signal is not HOLD
            if signal in ['BUY', 'SELL']:
                self.execute_trade(signal, current_price, current_date)

            # Update portfolio value
            portfolio_value = self.cash + (self.shares * current_price)

            self.portfolio_history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'shares': self.shares,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'signal': signal
            })

            # Progress logging every 1k timesteps
            if (idx + 1) % 1000 == 0:
                progress = ((idx + 1) / num_timesteps) * 100
                logging.info(f"  Processed {idx + 1}/{num_timesteps} timesteps ({progress:.1f}%)")

        # Calculate final metrics
        logging.info("Calculating performance metrics...")
        results = self.calculate_metrics()

        # Print summary
        logging.info(f"\n{'=' * 60}")
        logging.info("BACKTEST RESULTS")
        logging.info(f"{'=' * 60}")
        logging.info(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        logging.info(f"Total Return: {results['total_return']:.2f}%")
        logging.info(f"Expected Return (daily): {results['expected_return']:.4f}%")
        logging.info(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        logging.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logging.info(f"Win Rate: {results['win_rate']:.2f}%")
        logging.info(f"Total Trades: {results['total_trades']}")
        logging.info(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
        logging.info(f"{'=' * 60}\n")

        return results

    def calculate_metrics(self):
        """
        Calculate comprehensive performance metrics.

        Returns:
            dict: Performance metrics
        """
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # Basic metrics
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_balance - 1) * 100

        # Daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()

        # Expected return (mean daily return)
        expected_return = daily_returns.mean() * 100  # As percentage

        # Sharpe Ratio (annualized, assuming 252 trading days)
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max Drawdown
        cumulative_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100  # As percentage

        # Trade statistics
        total_trades = len(self.trades)

        if total_trades > 0:
            # Calculate profit/loss for each trade pair (buy-sell)
            profitable_trades = 0
            total_profit = 0

            buy_price = None
            for trade in self.trades:
                if trade['action'] == 'BUY':
                    buy_price = trade['price']
                elif trade['action'] == 'SELL' and buy_price is not None:
                    profit = (trade['price'] - buy_price) * trade['shares']
                    total_profit += profit
                    if profit > 0:
                        profitable_trades += 1
                    buy_price = None

            win_rate = (profitable_trades / (total_trades / 2)) * 100 if total_trades > 1 else 0
        else:
            win_rate = 0
            total_profit = 0

        # Buy and hold comparison
        first_price = portfolio_df['current_price'].iloc[0]
        last_price = portfolio_df['current_price'].iloc[-1]
        buy_hold_return = (last_price / first_price - 1) * 100

        # Total transaction costs paid
        total_fees = sum(trade.get('transaction_fee', 0) for trade in self.trades)

        return {
            'final_value': final_value,
            'total_return': total_return,
            'expected_return': expected_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'portfolio_history': portfolio_df,
            'trades': trades_df
        }


# ---------------------------
# Convenience function
# ---------------------------
def run_backtest(model, scaler, df, feature_columns, window_size=60,
                 initial_balance=10000, transaction_cost_pct=0.02,
                 threshold=0.01, start_idx=None, end_idx=None):
    """
    Convenience function to run a backtest with default settings.

    Args:
        model: Trained PyTorch model
        scaler: Fitted MinMaxScaler
        df: DataFrame with price data
        feature_columns: List of feature columns
        window_size: Prediction window size
        initial_balance: Starting cash
        transaction_cost_pct: Transaction cost (default 2%)
        threshold: Signal threshold (default 1%)
        start_idx: Optional start index
        end_idx: Optional end index

    Returns:
        dict: Backtest results
    """
    backtester = Backtester(
        model=model,
        scaler=scaler,
        df=df,
        feature_columns=feature_columns,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost_pct=transaction_cost_pct,
        threshold=threshold
    )

    return backtester.run(start_idx=start_idx, end_idx=end_idx)


# ---------------------------
# Example usage / testing
# ---------------------------
if __name__ == "__main__":
    print("backtest.py loaded successfully!")
    print("\nUsage example:")
    print("""
    from backtest import run_backtest

    results = run_backtest(
        model=trained_model,
        scaler=fitted_scaler,
        df=cleaned_dataframe,
        feature_columns=['close', 'RSI', 'MACD', 'MACD_Signal', 'SMA'],
        window_size=60,
        initial_balance=10000,
        transaction_cost_pct=0.02,  # 2% total cost
        threshold=0.01               # 1% signal threshold
    )

    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Expected Return: {results['expected_return']:.4f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    """)