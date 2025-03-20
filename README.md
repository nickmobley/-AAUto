# AAUto Trading System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

AAUto is a sophisticated automated trading and investment system that integrates multiple strategies, technical analysis, machine learning, and risk management to execute trades in financial markets.

## Features

- **Multi-strategy approach**: Combines trading, investments, and freelancing strategies
- **Technical Analysis**: Built-in indicators including RSI, MACD, EMA, and trend analysis
- **Machine Learning**: Price prediction and pattern recognition models
- **Risk Management**: Position sizing, stop-loss calculation, and drawdown management
- **News Analysis**: Market sentiment analysis from financial news
- **Performance Metrics**: Comprehensive tracking and visualization of trading performance

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Component Overview](#component-overview)
- [Best Practices](#best-practices)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AAUto
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the sample configuration file:
   ```bash
   cp config.sample.json config.json
   ```

5. Update the configuration with your API credentials and preferences (see [Configuration](#configuration) section)

## Configuration

The system is configured via the `config.json` file, which includes the following sections:

### API Configuration

```json
{
  "api": {
    "alpha_vantage": {
      "api_key": "YOUR_API_KEY_HERE",
      "base_url": "https://www.alphavantage.co/query",
      "rate_limit": {
        "calls_per_minute": 5,
        "calls_per_day": 500
      },
      "cache": {
        "enabled": true,
        "expiry_hours": 24
      }
    }
  }
}
```

### Risk Management Settings

```json
{
  "risk": {
    "max_position_size_percent": 5.0,
    "max_total_risk_percent": 20.0,
    "default_stop_loss_percent": 2.0,
    "default_take_profit_percent": 6.0,
    "max_drawdown_percent": 15.0
  }
}
```

### Trading Parameters

```json
{
  "trading": {
    "base_currency": "USD",
    "symbols": ["AAPL", "MSFT", "GOOG", "AMZN"],
    "default_timeframe": "1d",
    "trading_hours": {
      "start": "09:30",
      "end": "16:00",
      "timezone": "America/New_York"
    },
    "strategies": ["momentum", "reversal", "trend_following"]
  }
}
```

### Machine Learning Settings

```json
{
  "ml": {
    "model_type": "random_forest",
    "features": ["rsi", "macd", "ema", "volume", "sentiment"],
    "training": {
      "lookback_days": 365,
      "validation_split": 0.2,
      "retraining_frequency_days": 30
    }
  }
}
```

## Usage

### Basic Operation

To start the trading system:

```bash
python src/main.py
```

### Command-line Arguments

```bash
python src/main.py --config custom_config.json --debug --backtest 2023-01-01 2023-06-30
```

Available options:
- `--config`: Specify a custom configuration file (default: config.json)
- `--debug`: Enable debug logging
- `--backtest`: Run in backtest mode with start and end dates
- `--paper-trading`: Run in paper trading mode (no real trades)
- `--portfolio`: Show current portfolio status and exit

### Example Usage Scenarios

#### Paper Trading

```bash
python src/main.py --paper-trading
```

#### Backtesting a Strategy

```bash
python src/main.py --backtest 2022-01-01 2022-12-31 --strategy momentum
```

#### Running with Custom Risk Parameters

```bash
python src/main.py --risk-max-position 3.0 --risk-stop-loss 1.5
```

## Component Overview

### Alpha Vantage API (`src/api/alpha_vantage.py`)

Handles all interactions with the Alpha Vantage API, including rate limiting and response caching.

```python
from src.api.alpha_vantage import AlphaVantageAPI

# Example usage
api = AlphaVantageAPI(api_key="YOUR_KEY")
data = api.get_daily_adjusted("AAPL")
```

### Technical Analyzer (`src/analytics/technical.py`)

Calculates and interprets technical indicators for trading signals.

```python
from src.analytics.technical import TechnicalAnalyzer

# Example usage
analyzer = TechnicalAnalyzer()
rsi = analyzer.calculate_rsi(prices, period=14)
is_overbought = analyzer.is_overbought(rsi, threshold=70)
```

### Risk Manager (`src/risk/manager.py`)

Manages position sizing and risk parameters.

```python
from src.risk.manager import RiskManager

# Example usage
risk_manager = RiskManager(account_balance=10000)
position_size = risk_manager.calculate_position_size("AAPL", risk_percent=1.0)
```

### Machine Learning (`src/ml/predictor.py`)

Provides price prediction and pattern recognition.

```python
from src.ml.predictor import MachineLearning

# Example usage
ml = MachineLearning()
ml.train(historical_data)
prediction = ml.predict_price("AAPL", days_ahead=5)
```

### Trader (`src/core/trader.py`)

Core trading logic that integrates all components.

```python
from src.core.trader import Trader

# Example usage
trader = Trader(config_path="config.json")
trader.run()
```

## Best Practices

### API Key Security

- **Never commit your API keys** to version control
- Use environment variables or a secure vault for sensitive credentials
- Create a `.env` file for local development (add to `.gitignore`)

### Risk Management

- Start with small position sizes (1-2% of portfolio)
- Use stop losses for every trade
- Monitor drawdown and be prepared to stop trading if it exceeds your threshold
- Diversify across multiple symbols and strategies

### System Maintenance

- Regularly check logs for errors and warnings
- Back up your database and configuration regularly
- Monitor system resource usage, especially during high-frequency trading
- Periodically retrain machine learning models with fresh data

### Performance Analysis

- Review trading performance weekly and monthly
- Compare strategy performance against benchmarks (e.g., S&P 500)
- Analyze losing trades to identify patterns or improvements
- Consider adjusting parameters based on changing market conditions

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_api.py tests/test_technical.py

# Run with coverage report
pytest --cov=src
```

### Test Environment Setup

For testing, you can use the `--mock-api` flag to avoid making real API calls:

```bash
python src/main.py --paper-trading --mock-api
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the coding style guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

