#!/usr/bin/env python
"""
Command-line interface for the trading system.
Provides comprehensive command structure and system management capabilities.
"""

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Initialize logger
logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categories for CLI commands."""
    SYSTEM = "system"
    OPERATIONS = "operations"
    DEVELOPMENT = "development"
    DIAGNOSTICS = "diagnostics"


class CommandRegistry:
    """Registry for all CLI commands."""
    
    _commands: Dict[str, Dict[str, Callable]] = {
        category.value: {} for category in CommandCategory
    }
    
    @classmethod
    def register(cls, category: CommandCategory, name: str) -> Callable:
        """Decorator to register a command."""
        def decorator(func: Callable) -> Callable:
            cls._commands[category.value][name] = func
            return func
        return decorator
    
    @classmethod
    def get_command(cls, category: str, name: str) -> Optional[Callable]:
        """Get command by category and name."""
        return cls._commands.get(category, {}).get(name)
    
    @classmethod
    def get_commands(cls, category: Optional[str] = None) -> Dict[str, Dict[str, Callable]]:
        """Get all commands or commands for a specific category."""
        if category:
            return {category: cls._commands.get(category, {})}
        return cls._commands
    
    @classmethod
    def list_commands(cls) -> List[Tuple[str, str]]:
        """List all available commands."""
        commands = []
        for category, cmd_dict in cls._commands.items():
            for cmd_name in cmd_dict.keys():
                commands.append((category, cmd_name))
        return commands


class CLI:
    """Main CLI class for the trading system."""
    
    def __init__(self):
        """Initialize CLI with parsers and commands."""
        self.parser = self._create_parser()
        self._load_modules()
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all subcommands."""
        parser = argparse.ArgumentParser(
            description="Trading System Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging"
        )
        
        subparsers = parser.add_subparsers(dest="category", help="Command category")
        
        # System management commands
        system_parser = subparsers.add_parser(
            CommandCategory.SYSTEM.value, help="System management commands"
        )
        system_subparsers = system_parser.add_subparsers(dest="command", help="System command")
        
        system_subparsers.add_parser("start", help="Start the trading system")
        system_subparsers.add_parser("stop", help="Stop the trading system")
        system_subparsers.add_parser("restart", help="Restart the trading system")
        system_subparsers.add_parser("status", help="Get system status")
        
        # Operational commands
        operations_parser = subparsers.add_parser(
            CommandCategory.OPERATIONS.value, help="Operational commands"
        )
        operations_subparsers = operations_parser.add_subparsers(dest="command", help="Operation command")
        
        strategy_parser = operations_subparsers.add_parser("strategy", help="Strategy management")
        strategy_parser.add_argument("--list", action="store_true", help="List available strategies")
        strategy_parser.add_argument("--enable", type=str, help="Enable a strategy")
        strategy_parser.add_argument("--disable", type=str, help="Disable a strategy")
        
        risk_parser = operations_subparsers.add_parser("risk", help="Risk management")
        risk_parser.add_argument("--set-limit", type=float, help="Set risk limit")
        risk_parser.add_argument("--get-limits", action="store_true", help="Get current risk limits")
        
        # Development commands
        dev_parser = subparsers.add_parser(
            CommandCategory.DEVELOPMENT.value, help="Development commands"
        )
        dev_subparsers = dev_parser.add_subparsers(dest="command", help="Development command")
        
        test_parser = dev_subparsers.add_parser("test", help="Run tests")
        test_parser.add_argument("--unit", action="store_true", help="Run unit tests")
        test_parser.add_argument("--integration", action="store_true", help="Run integration tests")
        test_parser.add_argument("--system", action="store_true", help="Run system tests")
        
        build_parser = dev_subparsers.add_parser("build", help="Build system components")
        build_parser.add_argument("--all", action="store_true", help="Build all components")
        build_parser.add_argument("--component", type=str, help="Build specific component")
        
        # Diagnostic commands
        diag_parser = subparsers.add_parser(
            CommandCategory.DIAGNOSTICS.value, help="Diagnostic commands"
        )
        diag_subparsers = diag_parser.add_subparsers(dest="command", help="Diagnostic command")
        
        monitor_parser = diag_subparsers.add_parser("monitor", help="Monitor system metrics")
        monitor_parser.add_argument("--component", type=str, help="Monitor specific component")
        monitor_parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
        
        logs_parser = diag_subparsers.add_parser("logs", help="View system logs")
        logs_parser.add_argument("--component", type=str, help="Component to view logs for")
        logs_parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                               default="INFO", help="Log level")
        logs_parser.add_argument("--tail", type=int, default=100, help="Number of log lines to show")
        
        return parser
    
    def _load_modules(self) -> None:
        """Dynamically load command modules."""
        try:
            # Import all command modules to register their commands
            from src.cli import (
                system_commands,
                operations_commands,
                development_commands,
                diagnostic_commands,
            )
        except ImportError as e:
            logger.warning(f"Some command modules could not be imported: {e}")
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with the given arguments."""
        parsed_args = self.parser.parse_args(args)
        
        # Configure logging based on debug flag
        log_level = logging.DEBUG if parsed_args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        if not parsed_args.category:
            self.parser.print_help()
            return 0
        
        # Execute the selected command
        try:
            return self._execute_command(parsed_args)
        except Exception as e:
            logger.error(f"Command execution failed: {e}", exc_info=parsed_args.debug)
            return 1
    
    def _execute_command(self, args: argparse.Namespace) -> int:
        """Execute a command based on the parsed arguments."""
        if not hasattr(args, "command") or not args.command:
            # If only category is specified, print help for that category
            if args.category == CommandCategory.SYSTEM.value:
                self.parser._action_groups[2]._actions[0].print_help()
            elif args.category == CommandCategory.OPERATIONS.value:
                self.parser._action_groups[2]._actions[1].print_help()
            elif args.category == CommandCategory.DEVELOPMENT.value:
                self.parser._action_groups[2]._actions[2].print_help()
            elif args.category == CommandCategory.DIAGNOSTICS.value:
                self.parser._action_groups[2]._actions[3].print_help()
            return 0
        
        command_func = CommandRegistry.get_command(args.category, args.command)
        if not command_func:
            logger.error(f"Command not found: {args.category} {args.command}")
            return 1
        
        return command_func(args) or 0


# System Management Commands
@CommandRegistry.register(CommandCategory.SYSTEM, "start")
def start_system(args: argparse.Namespace) -> int:
    """Start the trading system."""
    from src.core.orchestration import SystemOrchestrator
    
    logger.info("Starting trading system...")
    try:
        orchestrator = SystemOrchestrator()
        asyncio.run(orchestrator.start())
        logger.info("Trading system started successfully")
        return 0
    except Exception as e:
        logger.error(f"Failed to start trading system: {e}")
        return 1


@CommandRegistry.register(CommandCategory.SYSTEM, "stop")
def stop_system(args: argparse.Namespace) -> int:
    """Stop the trading system."""
    from src.core.orchestration import SystemOrchestrator
    
    logger.info("Stopping trading system...")
    try:
        orchestrator = SystemOrchestrator()
        asyncio.run(orchestrator.stop())
        logger.info("Trading system stopped successfully")
        return 0
    except Exception as e:
        logger.error(f"Failed to stop trading system: {e}")
        return 1


@CommandRegistry.register(CommandCategory.SYSTEM, "restart")
def restart_system(args: argparse.Namespace) -> int:
    """Restart the trading system."""
    stop_result = stop_system(args)
    if stop_result != 0:
        return stop_result
    
    # Add a small delay to ensure clean shutdown
    import time
    time.sleep(2)
    
    return start_system(args)


@CommandRegistry.register(CommandCategory.SYSTEM, "status")
def system_status(args: argparse.Namespace) -> int:
    """Get system status."""
    from src.core.monitoring import SystemMonitor
    
    logger.info("Getting system status...")
    try:
        monitor = SystemMonitor()
        status = asyncio.run(monitor.get_status())
        
        # Pretty print the status information
        print(json.dumps(status, indent=2))
        return 0
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return 1


# Operational Control Commands
@CommandRegistry.register(CommandCategory.OPERATIONS, "strategy")
def manage_strategy(args: argparse.Namespace) -> int:
    """Manage trading strategies."""
    from src.strategy import StrategyManager
    
    try:
        manager = StrategyManager()
        
        if args.list:
            strategies = asyncio.run(manager.list_strategies())
            print("Available strategies:")
            for strategy in strategies:
                print(f"- {strategy['name']}: {strategy['description']} [{'ENABLED' if strategy['enabled'] else 'DISABLED'}]")
            return 0
        
        if args.enable:
            asyncio.run(manager.enable_strategy(args.enable))
            print(f"Strategy '{args.enable}' enabled successfully")
            return 0
        
        if args.disable:
            asyncio.run(manager.disable_strategy(args.disable))
            print(f"Strategy '{args.disable}' disabled successfully")
            return 0
        
        # If no specific action is provided, show help
        print("Please specify an action: --list, --enable <strategy>, or --disable <strategy>")
        return 1
    except Exception as e:
        logger.error(f"Strategy management failed: {e}")
        return 1


@CommandRegistry.register(CommandCategory.OPERATIONS, "risk")
def manage_risk(args: argparse.Namespace) -> int:
    """Manage risk settings."""
    from src.risk.portfolio import RiskManager
    
    try:
        manager = RiskManager()
        
        if args.set_limit is not None:
            asyncio.run(manager.set_risk_limit(args.set_limit))
            print(f"Risk limit set to {args.set_limit}")
            return 0
        
        if args.get_limits:
            limits = asyncio.run(manager.get_risk_limits())
            print("Current risk limits:")
            print(json.dumps(limits, indent=2))
            return 0
        
        # If no specific action is provided, show help
        print("Please specify an action: --set-limit <value> or --get-limits")
        return 1
    except Exception as e:
        logger.error(f"Risk management failed: {e}")
        return 1


# Development and Diagnostic Commands
@CommandRegistry.register(CommandCategory.DEVELOPMENT, "test")
def run_tests(args: argparse.Namespace) -> int:
    """Run system tests."""
    from src.tests import TestRunner
    
    try:
        runner = TestRunner()
        
        if args.unit:
            results = asyncio.run(runner.run_unit_tests())
            print(f"Unit tests completed: {results['passed']}/{results['total']} passed")
            return 0 if results["passed"] == results["total"] else 1
        
        if args.integration:
            results = asyncio.run(runner.run_integration_tests())
            print(f"Integration tests completed: {results['passed']}/{results['total']} passed")
            return 0 if results["passed"] == results["total"] else 1
        
        if args.system:
            results = asyncio.run(runner.run_system_tests())
            print(f"System tests completed: {results['passed']}/{results['total']} passed")
            return 0 if results["passed"] == results["total"] else 1
        
        # If no specific tests are specified, run all tests
        all_results = asyncio.run(runner.run_all_tests())
        print("All tests completed:")
        print(f"- Unit tests: {all_results['unit']['passed']}/{all_results['unit']['total']} passed")
        print(f"- Integration tests: {all_results['integration']['passed']}/{all_results['integration']['total']} passed")
        print(f"- System tests: {all_results['system']['passed']}/{all_results['system']['total']} passed")
        
        # Return success only if all tests passed
        return 0 if (all_results['unit']['passed'] == all_results['unit']['total'] and
                     all_results['integration']['passed'] == all_results['integration']['total'] and
                     all_results['system']['passed'] == all_results['system']['total']) else 1
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


@CommandRegistry.register(CommandCategory.DEVELOPMENT, "build")
def build_components(args: argparse.Namespace) -> int:
    """Build system components."""
    from src.build import BuildManager
    
    try:
        manager = BuildManager()
        
        if args.all:
            results = asyncio.run(manager.build_all())
            print("Build completed for all components")
            for component, status in results.items():

