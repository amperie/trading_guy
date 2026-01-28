"""
Split-period backtesting engine for train/validation testing.

Runs two sequential backtests on different time periods to evaluate strategy
performance on out-of-sample data.
"""
from trading.engines.base_engine import BaseEngine
from trading.engines.backtest_engine import BacktestingEngine
from trading.engines.analysis_engine import AnalysisEngine
from trading.data_providers.data_provider import DataProvider
from trading.core.algorithm import Algorithm
from trading.core.portfolio import Portfolio
from trading.core.order_manager import OrderManager
from typing import Optional, Dict, Any
from utils.logger import Logger
from dataclasses import asdict
from datetime import datetime
import copy

logger = Logger().get_logger(__name__)


class SplitPeriodBacktestEngine(BaseEngine):
    """
    Runs two sequential backtests on split time periods for train/validation testing.

    This engine enables out-of-sample testing by splitting historical data into two periods:
    - Training period: start_date to cutoff_date
    - Validation period: cutoff_date to end_date

    Both periods start with the same initial conditions (cash, no positions) and use
    the same algorithm/portfolio parameters. Results are logged to MLflow with the
    validation run including training metrics for easy comparison.
    """

    def __init__(
        self,
        cfg: dict = None,
        dp: DataProvider = None,
        al: Algorithm = None,
        om: OrderManager = None,
        pf: Portfolio = None,
        start_date: str = None,
        cutoff_date: str = None,
        end_date: str = None
    ):
        """
        Initialize split-period backtest engine.

        Args:
            cfg: Engine configuration (must include experiment_name, run_name)
            dp: Data provider (will be re-initialized with date filters)
            al: Algorithm (will be re-initialized for validation period)
            om: Order manager (will be re-initialized for validation period)
            pf: Portfolio (will be re-initialized for validation period)
            start_date: Start of training period (format: 'YYYY-MM-DD')
            cutoff_date: End of training / start of validation (format: 'YYYY-MM-DD')
            end_date: End of validation period (format: 'YYYY-MM-DD')
        """
        super().__init__(cfg=cfg, dp=dp, al=al, om=om, pf=pf)

        # Store date parameters
        self.start_date = start_date
        self.cutoff_date = cutoff_date
        self.end_date = end_date

        # Validate dates
        self._validate_dates()

        # Extract MLflow configuration
        self.experiment_name = cfg.get('experiment_name', 'Split Period Backtest') if cfg else 'Split Period Backtest'
        self.run_name = cfg.get('run_name', 'Backtest') if cfg else 'Backtest'
        self.description = cfg.get('description', '') if cfg else ''

        # Store original configs for re-initialization
        self.original_dp_cfg = copy.deepcopy(dp.cfg) if dp else {}
        self.original_al_cfg = copy.deepcopy(al.cfg) if (al and hasattr(al, 'cfg')) else {}
        self.original_pf_cfg = copy.deepcopy(pf.cfg) if (pf and hasattr(pf, 'cfg')) else {}
        self.starting_cash = pf.cash if pf else 0.0

    def _validate_dates(self):
        """Ensure dates are in correct chronological order."""
        if not all([self.start_date, self.cutoff_date, self.end_date]):
            raise ValueError("start_date, cutoff_date, and end_date are all required")

        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        cutoff = datetime.strptime(self.cutoff_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        if not (start < cutoff < end):
            raise ValueError(
                f"Dates must be in order: start < cutoff < end. "
                f"Got: {self.start_date} < {self.cutoff_date} < {self.end_date}"
            )

    def run(self):
        """
        Execute both training and validation backtests.

        Returns:
            Dictionary with:
            - training: Results from training period
            - validation: Results from validation period
        """
        logger.info("=" * 80)
        logger.info("SPLIT PERIOD BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Training Period:   {self.start_date} to {self.cutoff_date}")
        logger.info(f"Validation Period: {self.cutoff_date} to {self.end_date}")
        logger.info("=" * 80)

        # Run training period backtest
        try:
            train_results = self._run_training_period()
        except Exception as e:
            logger.error(f"Training period failed: {e}", exc_info=True)
            raise

        # Run validation period backtest
        try:
            val_results = self._run_validation_period(train_results)
        except Exception as e:
            logger.error(f"Validation period failed: {e}", exc_info=True)
            raise

        logger.info("=" * 80)
        logger.info("SPLIT PERIOD BACKTEST COMPLETE")
        logger.info("=" * 80)

        return {
            'training': train_results,
            'validation': val_results
        }

    def _run_training_period(self) -> Dict[str, Any]:
        """
        Run backtest on training period (start_date to cutoff_date).

        Returns:
            Dictionary with:
            - metrics: PerformanceMetrics object
            - trades: List of Trade objects
            - other analysis results
        """
        logger.info("Starting TRAINING period backtest...")
        logger.info(f"Period: {self.start_date} to {self.cutoff_date}")

        # Create fresh DataProvider with training date range
        train_dp_cfg = copy.deepcopy(self.original_dp_cfg)
        train_dp_cfg['start_date'] = self.start_date
        train_dp_cfg['end_date'] = self.cutoff_date

        # Import DataProvider class dynamically based on original type
        dp_class = type(self.dp)
        train_dp = dp_class(train_dp_cfg)

        # Create fresh OrderManager
        om_class = type(self.om)
        train_om = om_class()

        # Create fresh Algorithm
        al_class = type(self.al)
        train_al = al_class(self.original_al_cfg)

        # Create fresh Portfolio with same starting conditions
        pf_class = type(self.pf)
        train_pf = pf_class(
            self.original_pf_cfg,
            train_om,
            self.starting_cash,
            {},  # No starting positions
            keep_history=True  # Need history for analysis
        )

        # Run backtest
        logger.info("Running training backtest engine...")
        train_engine = BacktestingEngine({}, train_dp, train_al, train_om, train_pf)
        train_engine.run()

        # Analyze results
        logger.info("Analyzing training period results...")
        analysis_engine = AnalysisEngine(train_pf, train_om)

        # Extract parameters for logging
        parameters = self._extract_all_parameters(train_al, train_pf, train_dp)
        parameters['period'] = 'training'
        parameters['start_date'] = self.start_date
        parameters['cutoff_date'] = self.cutoff_date

        # Run full analysis and log to MLflow
        train_results = analysis_engine.run_full_analysis(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            description=f"{self.description} [Training: {self.start_date} to {self.cutoff_date}]",
            parameters=parameters,
            log_to_mlflow=True,
            save_charts_locally=False,
            save_report_locally=False
        )

        logger.info("Training period complete!")
        logger.info(f"  Total Return: {train_results['metrics'].total_return_pct:.2f}%")
        logger.info(f"  Sharpe Ratio: {train_results['metrics'].sharpe_ratio:.2f}")
        logger.info(f"  Total Trades: {train_results['metrics'].total_trades}")
        logger.info("")

        return train_results

    def _run_validation_period(self, train_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest on validation period (cutoff_date to end_date).
        Logs to MLflow with "_validation" suffix and includes training metrics.

        Args:
            train_results: Results from training period (includes metrics)

        Returns:
            Dictionary with validation results
        """
        logger.info("Starting VALIDATION period backtest...")
        logger.info(f"Period: {self.cutoff_date} to {self.end_date}")

        # Create fresh DataProvider with validation date range
        val_dp_cfg = copy.deepcopy(self.original_dp_cfg)
        val_dp_cfg['start_date'] = self.cutoff_date
        val_dp_cfg['end_date'] = self.end_date

        # Import DataProvider class dynamically
        dp_class = type(self.dp)
        val_dp = dp_class(val_dp_cfg)

        # Create fresh OrderManager
        om_class = type(self.om)
        val_om = om_class()

        # Create fresh Algorithm (same config as training)
        al_class = type(self.al)
        val_al = al_class(self.original_al_cfg)

        # Create fresh Portfolio with SAME starting conditions as training
        pf_class = type(self.pf)
        val_pf = pf_class(
            self.original_pf_cfg,
            val_om,
            self.starting_cash,  # Fresh start with same initial cash
            {},  # No starting positions
            keep_history=True
        )

        # Run backtest
        logger.info("Running validation backtest engine...")
        val_engine = BacktestingEngine({}, val_dp, val_al, val_om, val_pf)
        val_engine.run()

        # Analyze results
        logger.info("Analyzing validation period results...")
        analysis_engine = AnalysisEngine(val_pf, val_om)

        # Extract parameters (same as training)
        parameters = self._extract_all_parameters(val_al, val_pf, val_dp)
        parameters['period'] = 'validation'
        parameters['cutoff_date'] = self.cutoff_date
        parameters['end_date'] = self.end_date

        # Add training metrics with "train_" prefix
        train_metrics = train_results['metrics']
        training_metrics_dict = self._prefix_metrics(train_metrics, 'train_')
        parameters.update(training_metrics_dict)

        # Run full analysis and log to MLflow with "_validation" suffix
        val_results = analysis_engine.run_full_analysis(
            experiment_name=self.experiment_name,
            run_name=f"{self.run_name}_validation",  # Append "_validation"
            description=f"{self.description} [Validation: {self.cutoff_date} to {self.end_date}]",
            parameters=parameters,
            log_to_mlflow=True,
            save_charts_locally=False,
            save_report_locally=False
        )

        logger.info("Validation period complete!")
        logger.info(f"  Total Return: {val_results['metrics'].total_return_pct:.2f}%")
        logger.info(f"  Sharpe Ratio: {val_results['metrics'].sharpe_ratio:.2f}")
        logger.info(f"  Total Trades: {val_results['metrics'].total_trades}")
        logger.info("")

        # Print comparison
        logger.info("COMPARISON:")
        logger.info(f"  Training Return:   {train_metrics.total_return_pct:>8.2f}%")
        logger.info(f"  Validation Return: {val_results['metrics'].total_return_pct:>8.2f}%")
        logger.info(f"  Training Sharpe:   {train_metrics.sharpe_ratio:>8.2f}")
        logger.info(f"  Validation Sharpe: {val_results['metrics'].sharpe_ratio:>8.2f}")

        return val_results

    def _extract_all_parameters(
        self,
        algorithm: Algorithm,
        portfolio: Portfolio,
        data_provider: DataProvider
    ) -> Dict[str, Any]:
        """
        Extract all configuration parameters from components.

        Args:
            algorithm: Algorithm instance
            portfolio: Portfolio instance
            data_provider: DataProvider instance

        Returns:
            Dictionary with prefixed parameters (alg_, pf_, dp_)
        """
        params = {}

        # Algorithm parameters
        if hasattr(algorithm, 'cfg') and algorithm.cfg:
            params.update({f'alg_{k}': v for k, v in algorithm.cfg.items()})

        # Portfolio parameters
        if hasattr(portfolio, 'cfg') and portfolio.cfg:
            params.update({f'pf_{k}': v for k, v in portfolio.cfg.items()})

        # Data provider parameters
        if hasattr(data_provider, 'cfg') and data_provider.cfg:
            params.update({f'dp_{k}': v for k, v in data_provider.cfg.items()})

        # Add starting conditions
        params['starting_cash'] = self.starting_cash

        return params

    def _prefix_metrics(
        self,
        metrics: 'PerformanceMetrics',
        prefix: str
    ) -> Dict[str, Any]:
        """
        Convert PerformanceMetrics to dict with prefixed keys.

        Args:
            metrics: PerformanceMetrics dataclass
            prefix: Prefix to add (e.g., "train_")

        Returns:
            Dictionary with prefixed metric names
        """
        metrics_dict = asdict(metrics)
        prefixed = {}

        for key, value in metrics_dict.items():
            # Skip None values
            if value is not None:
                prefixed[f"{prefix}{key}"] = value

        return prefixed

    def on_tick(self, tick):
        """Not used - individual BacktestingEngines handle ticks."""
        pass

    def finalize(self):
        """Not used - individual BacktestingEngines handle finalization."""
        pass
