"""
Chart Metrics Management Class

This module provides a centralized way to manage and define chart metrics
for the Hunter-Prey simulation visualization.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ChartMetrics:
    """
    Manages chart metrics for the Hunter-Prey simulation visualization.
    
    This class centralizes the definition and management of metrics that are
    displayed in the simulation charts, making it easier to add, remove, or

    modify metrics without changing the main simulation code.    """

    
    def __init__(self):
        """Initialize the ChartMetrics with default metrics."""
        self._metrics = [
            "NashQHunters",
            "NashQPreys", 
            "NashQHunterKills",
            "AvgNashQHunterReward",

            "AvgNashQPreyReward",
            "MinimaxHunters",
            "MinimaxPreys",
            "MinimaxHunterKills", 
            "AvgMinimaxHunterReward",
            "AvgMinimaxPreyReward"

        ]
        logger.info(f"ChartMetrics initialized with {len(self._metrics)} metrics")
    
    @property
    def metrics(self) -> List[str]:
        """
        Get the current list of chart metrics.
        
        Returns:
            List of metric names as strings
        """
        return self._metrics.copy()
    
    def add_metric(self, metric_name: str) -> None:
        """
        Add a new metric to the chart metrics list.
        
        Args:
            metric_name: Name of the metric to add
            
        Raises:
            ValueError: If metric already exists
        """
        if metric_name in self._metrics:
            raise ValueError(f"Metric '{metric_name}' already exists")
        
        self._metrics.append(metric_name)
        logger.info(f"Added metric: {metric_name}")
    
    def remove_metric(self, metric_name: str) -> None:
        """
        Remove a metric from the chart metrics list.
        
        Args:
            metric_name: Name of the metric to remove
            
        Raises:
            ValueError: If metric doesn't exist
        """
        if metric_name not in self._metrics:
            raise ValueError(f"Metric '{metric_name}' not found")
        
        self._metrics.remove(metric_name)
        logger.info(f"Removed metric: {metric_name}")
    
    def has_metric(self, metric_name: str) -> bool:
        """
        Check if a metric exists in the current metrics list.
        
        Args:
            metric_name: Name of the metric to check
            
        Returns:
            True if metric exists, False otherwise
        """
        return metric_name in self._metrics
    
    def get_nash_q_metrics(self) -> List[str]:
        """
        Get only the Nash Q-related metrics.
        
        Returns:
            List of Nash Q-specific metric names
        """
        nash_q_metrics = [
            metric for metric in self._metrics 
            if "NashQ" in metric
        ]
        return nash_q_metrics
    

    def get_minimax_metrics(self) -> List[str]:
        """
        Get only the Minimax-related metrics.
        
        Returns:
            List of Minimax-specific metric names
        """
        minimax_metrics = [
            metric for metric in self._metrics 
            if "Minimax" in metric
        ]
        return minimax_metrics
    

    def get_hunter_metrics(self) -> List[str]:
        """
        Get only the hunter-related metrics.
        
        Returns:
            List of hunter-specific metric names
        """
        hunter_metrics = [
            metric for metric in self._metrics 
            if "Hunter" in metric
        ]
        return hunter_metrics
    
    def get_prey_metrics(self) -> List[str]:
        """
        Get only the prey-related metrics.
        
        Returns:
            List of prey-specific metric names
        """
        prey_metrics = [
            metric for metric in self._metrics 
            if "Prey" in metric
        ]
        return prey_metrics
    
    def get_reward_metrics(self) -> List[str]:
        """
        Get only the reward-related metrics.
        
        Returns:
            List of reward-specific metric names
        """
        reward_metrics = [
            metric for metric in self._metrics 
            if "Reward" in metric
        ]
        return reward_metrics
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current metrics.
        
        Returns:
            Dictionary containing metrics statistics and categorization
        """
        return {
            "total_metrics": len(self._metrics),
            "all_metrics": self.metrics,
            "nash_q_metrics": self.get_nash_q_metrics(),

            "minimax_metrics": self.get_minimax_metrics(),

            "hunter_metrics": self.get_hunter_metrics(),
            "prey_metrics": self.get_prey_metrics(),
            "reward_metrics": self.get_reward_metrics(),
            "categories": {
                "nash_q": len(self.get_nash_q_metrics()),

                "minimax": len(self.get_minimax_metrics()),
                "hunter": len(self.get_hunter_metrics()),
                "prey": len(self.get_prey_metrics()),
                "reward": len(self.get_reward_metrics())
            }        }

    
    def reset_to_default(self) -> None:
        """Reset metrics to the default configuration."""
        self._metrics = [
            "NashQHunters",
            "NashQPreys", 
            "NashQHunterKills",
            "AvgNashQHunterReward",

            "AvgNashQPreyReward",
            "MinimaxHunters",
            "MinimaxPreys",
            "MinimaxHunterKills", 
            "AvgMinimaxHunterReward",
            "AvgMinimaxPreyReward"

        ]
        logger.info("ChartMetrics reset to default configuration")
    
    def __len__(self) -> int:
        """Return the number of metrics."""
        return len(self._metrics)
    
    def __contains__(self, metric_name: str) -> bool:
        """Check if a metric is in the metrics list using 'in' operator."""
        return metric_name in self._metrics
    
    def __iter__(self):
        """Allow iteration over the metrics."""
        return iter(self._metrics)
    
    def __str__(self) -> str:
        """String representation of the ChartMetrics."""
        return f"ChartMetrics({len(self._metrics)} metrics: {', '.join(self._metrics)})"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"ChartMetrics(metrics={self._metrics})"


# Create a singleton instance for use throughout the application
chart_metrics = ChartMetrics()
