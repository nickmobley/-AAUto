#!/usr/bin/env python3
"""
AAUto - Automated Trading System Main Entry Point

This module serves as the main entry point for the AAUto trading system.
It handles configuration loading, system initialization, the main trading loop,
and graceful shutdown operations.
"""

import os
import sys
import time
import signal
import logging
import argparse
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import configparser
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the components
from src.api.alpha_vantage import AlphaVantageAPI
from src.analytics.technical import TechnicalAnalyzer
from src.risk.manager import RiskManager
from src.ml.predictor import MachineLearning
from src.core.trader import Trader

# Global variables
running = True
trader = None
logger = None


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup and configure the logging system.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file. If None, logging will be to console only.
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("AAUto")
    
    # Set level
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    
    config = {}
    
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.endswith(('.ini', '.cfg')):
            parser = configparser.ConfigParser()
            parser.read(config_path)
            
            # Convert the ConfigParser object to a dictionary
            config = {
                section: dict(parser.items(section)) 
                for section in parser.sections()
            }
        else:
            logger.error(f"Unsupported config file format: {config_path}")
            raise ValueError(f"Unsupported config file format: {config_path}")
            
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise
    
    # Override configuration with environment variables
    load_dotenv()
    
    # Override API key with environment variable if present
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if api_key:
        if isinstance(config, dict):
            if 'api' not in config:
                config['api'] = {}
            config['api']['api_key'] = api_key
        else:
            logger.warning("Config is not a dictionary, cannot override API key")
    
    return config


def init_components(config: Dict[str, Any]):
    """
    Initialize all system components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of initialized components
    """
    logger.info("Initializing system components")
    
    try:
        # Extract configuration parameters
        api_config = config.get('api', {})
        risk_config = config.get('risk', {})
        trading_config = config.get('trading', {})
        ml_config = config.get('ml', {})
        
        # Initialize API client
        api_key = api_config.get('api_key', os.environ.get('ALPHA_VANTAGE_API_KEY'))
        if not api_key:
            raise ValueError("Alpha Vantage API key not found in config or environment variables")
        
        api_client = AlphaVantageAPI(api_key=api_key, 
                                  cache_dir=api_config.get('cache_dir', './cache'),
                                  rate_limit=int(api_config.get('rate_limit', 5)))
        
        # Initialize technical analyzer
        tech_analyzer = TechnicalAnalyzer(api_client=api_client)
        
        # Initialize risk manager
        risk_manager = RiskManager(
            max_position_size=float(risk_config.get('max_position_size', 0.1)),
            max_risk_per_trade=float(risk_config.get('max_risk_per_trade', 0.02)),
            max_drawdown=float(risk_config.get('max_drawdown', 0.15)),
            position_sizing_method=risk_config.get('position_sizing_method', 'percentage')
        )
        
        # Initialize machine learning module
        ml_predictor = MachineLearning(
            model_path=ml_config.get('model_path', './models'),
            prediction_interval=int(ml_config.get('prediction_interval', 5)),
            confidence_threshold=float(ml_config.get('confidence_threshold', 0.7))
        )
        
        # Initialize trader
        trader_instance = Trader(
            api_client=api_client,
            technical_analyzer=tech_analyzer,
            risk_manager=risk_manager,
            ml_predictor=ml_predictor,
            symbols=trading_config.get('symbols', ['AAPL', 'MSFT', 'GOOGL']),
            trade_interval=int(trading_config.get('trade_interval', 60)),
            strategy=trading_config.get('strategy', 'combined')
        )
        
        return trader_instance
    
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


def signal_handler(sig, frame):
    """
    Handle termination signals for graceful shutdown.
    """
    global running, trader, logger
    
    signal_name = 'SIGTERM' if sig == signal.SIGTERM else 'SIGINT'
    logger.info(f"Received {signal_name}. Initiating graceful shutdown...")
    
    running = False
    
    if trader:
        try:
            # Close all open positions
            trader.close_all_positions()
            logger.info("All positions closed")
            
            # Save the current state
            trader.save_state()
            logger.info("Trader state saved")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


def main_loop(trader_instance, interval=60):
    """
    The main trading loop.
    
    Args:
        trader_instance: The initialized Trader object
        interval: Time interval between iterations in seconds
    """
    global running, logger
    
    logger.info("Starting main trading loop")
    
    while running:
        try:
            start_time = time.time()
            
            # Update market data
            trader_instance.update_market_data()
            
            # Analyze the market
            trader_instance.analyze_market()
            
            # Check and manage existing positions
            trader_instance.manage_positions()
            
            # Find and execute new opportunities
            trader_instance.find_opportunities()
            
            # Calculate and log performance metrics
            metrics = trader_instance.calculate_performance_metrics()
            logger.info(f"Performance metrics: {metrics}")
            
            # Calculate remaining time to sleep
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            
            if running and sleep_time > 0:
                logger.debug(f"Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            running = False
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.debug(traceback.format_exc())
            logger.info(f"Sleeping for {interval} seconds before retry")
            time.sleep(interval)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='AAUto - Automated Trading System')
    
    parser.add_argument(
        '-c', '--config', 
        default='config.json',
        help='Path to the configuration file'
    )
    
    parser.add_argument(
        '-l', '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    parser.add_argument(
        '--log-file',
        default='aauto.log',
        help='Path to the log file'
    )
    
    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='Disable logging to file'
    )
    
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=60,
        help='Trading loop interval in seconds'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    log_file = None if args.no_log_file else args.log_file
    logger = setup_logging(args.log_level, log_file)
    
    try:
        # Display startup banner
        logger.info("=" * 50)
        logger.info(f"AAUto Trading System v1.0.0")
        logger.info(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Load configuration
        config = load_config(args.config)
        
        # Initialize components
        trader = init_components(config)
        
        # Start the main loop
        main_loop(trader, args.interval)
        
    except Exception as e:
        if logger:
            logger.critical(f"Fatal error: {str(e)}")
            logger.debug(traceback.format_exc())
        else:
            print(f"Fatal error: {str(e)}")
            traceback.print_exc()
        sys.exit(1)
        
    finally:
        if logger:
            logger.info("AAUto system shutdown complete")

