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
    modify metrics without changing the main simulation code.    
    """

    
    def __init__(self):
        """Initialize the ChartMetrics with default metrics."""        # All available metrics
        self._all_metrics = [
            # NashQ metrics
            "NashQHunters",
            "NashQPreys", 
            "NashQHunterKills",
            "AvgNashQHunterReward",
            "AvgNashQPreyReward",
            
            # Minimax metrics
            "MinimaxHunters",
            "MinimaxPreys",
            "MinimaxHunterKills", 
            "AvgMinimaxHunterReward",
            "AvgMinimaxPreyReward",
            
            # MinimaxQ metrics
            "MinimaxQHunters",
            "MinimaxQPreys",
            "MinimaxQHunterKills",
            "AvgMinimaxQHunterReward",
            "AvgMinimaxQPreyReward",
            
            # Basic agent metrics
            "Hunters",
            "Preys",
            "RandomHunterKills",
            "GreedyHunterKills",
            "AvgHunterReward",
            "AvgPreyReward"
        ]
        
        # Active metrics (filtered based on current agents)
        self._active_metrics = self._all_metrics.copy()
        
        logger.info(f"ChartMetrics initialized with {len(self._active_metrics)} metrics")
    @property
    def metrics(self) -> List[str]:
        """
        Get the current list of active chart metrics.
        
        Returns:
            List of active metric names as strings
        """
        return self._active_metrics.copy()
    
    @property
    def all_metrics(self) -> List[str]:
        """
        Get the complete list of all available chart metrics.
        
        Returns:
            List of all metric names as strings
        """
        return self._all_metrics.copy()
    
    def add_metric(self, metric_name: str) -> None:
        """
        Add a new metric to the chart metrics list.
        
        Args:
            metric_name: Name of the metric to add
            
        Raises:
            ValueError: If metric already exists
        """
        if metric_name in self._all_metrics:
            raise ValueError(f"Metric '{metric_name}' already exists")
        
        self._all_metrics.append(metric_name)
        self._active_metrics.append(metric_name)
        logger.info(f"Added metric: {metric_name}")
    
    def remove_metric(self, metric_name: str) -> None:
        """
        Remove a metric from the chart metrics list.
        
        Args:
            metric_name: Name of the metric to remove
            
        Raises:
            ValueError: If metric doesn't exist
        """
        if metric_name not in self._all_metrics:
            raise ValueError(f"Metric '{metric_name}' not found")
        
        self._all_metrics.remove(metric_name)
        if metric_name in self._active_metrics:
            self._active_metrics.remove(metric_name)
        logger.info(f"Removed metric: {metric_name}")
    
    def has_metric(self, metric_name: str) -> bool:
        """
        Check if a metric exists in the current active metrics list.
        
        Args:
            metric_name: Name of the metric to check
            
        Returns:
            True if metric exists in active metrics, False otherwise        """
        return metric_name in self._active_metrics
    
    def get_nash_q_metrics(self) -> List[str]:
        """
        Get only the Nash Q-related metrics from active metrics.
        
        Returns:
            List of Nash Q-specific metric names
        """
        nash_q_metrics = [
            metric for metric in self._active_metrics 
            if "NashQ" in metric
        ]
        return nash_q_metrics
    def get_minimax_metrics(self) -> List[str]:
        """
        Get only the Minimax-related metrics from active metrics.
        
        Returns:
            List of Minimax-specific metric names
        """
        minimax_metrics = [
            metric for metric in self._active_metrics 
            if "Minimax" in metric
        ]
        return minimax_metrics
    
    def get_minimax_q_metrics(self) -> List[str]:
        """
        Get only the Minimax Q-related metrics from active metrics.
        
        Returns:
            List of Minimax Q-specific metric names
        """
        minimax_q_metrics = [
            metric for metric in self._active_metrics 
            if "MinimaxQ" in metric
        ]
        return minimax_q_metrics
    def get_hunter_metrics(self) -> List[str]:
        """
        Get only the hunter-related metrics from active metrics.
        
        Returns:
            List of hunter-specific metric names
        """
        hunter_metrics = [
            metric for metric in self._active_metrics 
            if "Hunter" in metric
        ]
        return hunter_metrics
    
    def get_prey_metrics(self) -> List[str]:
        """
        Get only the prey-related metrics from active metrics.
        
        Returns:
            List of prey-specific metric names
        """
        prey_metrics = [
            metric for metric in self._active_metrics 
            if "Prey" in metric
        ]
        return prey_metrics
    
    def get_reward_metrics(self) -> List[str]:
        """
        Get only the reward-related metrics from active metrics.
        
        Returns:
            List of reward-specific metric names
        """
        reward_metrics = [
            metric for metric in self._active_metrics 
            if "Reward" in metric
        ]
        return reward_metrics
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current active metrics.
        
        Returns:
            Dictionary containing metrics statistics and categorization
        """     
        
        return {
            "total_metrics": len(self._active_metrics),
            "active_metrics": self.metrics,
            "all_metrics": self.all_metrics,
            "nash_q_metrics": self.get_nash_q_metrics(),
            "minimax_metrics": self.get_minimax_metrics(),
            "minimax_q_metrics": self.get_minimax_q_metrics(),
            "hunter_metrics": self.get_hunter_metrics(),
            "prey_metrics": self.get_prey_metrics(),
            "reward_metrics": self.get_reward_metrics(),
            "categories": {
                "nash_q": len(self.get_nash_q_metrics()),
                "minimax": len(self.get_minimax_metrics()),
                "minimax_q": len(self.get_minimax_q_metrics()),
                "hunter": len(self.get_hunter_metrics()),
                "prey": len(self.get_prey_metrics()),
                "reward": len(self.get_reward_metrics())
            }
        }

    def reset_to_default(self) -> None:
        """Reset metrics to the default configuration."""
        self._active_metrics = self._all_metrics.copy()
        logger.info("ChartMetrics reset to default configuration")
    def update_active_metrics(self, model) -> None:
        """
        Update active metrics based on agent types present in the model.
        
        Args:
            model: The HunterPreyModel instance to check for agent types
        """
        # Start with empty list and only add metrics for agent types that exist
        self._active_metrics = []
          # Check for specific agent types
        has_base_preys = False
        has_random_hunters = False
        has_greedy_hunters = False
        has_nash_q_hunters = False
        has_minimax_hunters = False
        has_nash_q_preys = False
        has_minimax_preys = False
        has_minimax_q_hunters = False
        has_minimax_q_preys = False
        for agent in model.agents:
            agent_class = agent.__class__.__name__
            if agent_class == "RandomHunter":
                has_random_hunters = True
            elif agent_class == "Prey":
                has_base_preys = True
            elif agent_class == "GreedyHunter":
                has_greedy_hunters = True
            elif agent_class == "NashQHunter":
                has_nash_q_hunters = True
            elif agent_class == "MinimaxHunter":
                has_minimax_hunters = True
            elif agent_class == "NashQPrey":
                has_nash_q_preys = True
            elif agent_class == "MinimaxPrey":
                has_minimax_preys = True        
            elif agent_class == "MinimaxQHunter":
                has_minimax_q_hunters = True
            elif agent_class == "MinimaxQPrey":
                has_minimax_q_preys = True
        # Add base metrics only if base agents are present
        if has_random_hunters:
            self._active_metrics.append("Hunters")
        if has_base_preys:
            self._active_metrics.append("Preys")
        
        # Add metrics based on agent types present
        if has_random_hunters:
            self._active_metrics.append("RandomHunterKills")
            self._active_metrics.append("AvgHunterReward")
        
        if has_greedy_hunters:
            self._active_metrics.append("GreedyHunterKills")
            self._active_metrics.append("AvgHunterReward")
        
        if has_nash_q_hunters:
            self._active_metrics.extend([
                "NashQHunters",
                "NashQHunterKills",
                "AvgNashQHunterReward"
            ])
        
        if has_minimax_hunters:
            self._active_metrics.extend([
                "MinimaxHunters",
                "MinimaxHunterKills",
                "AvgMinimaxHunterReward"
            ])
        
        if has_nash_q_preys:
            self._active_metrics.extend([
                "NashQPreys",
                "AvgNashQPreyReward"
            ])
        
        if has_minimax_preys:
            self._active_metrics.extend([
                "MinimaxPreys",
                "AvgMinimaxPreyReward"
            ])
        
        if has_minimax_q_hunters:
            self._active_metrics.extend([
                "MinimaxQHunters",
                "MinimaxQHunterKills",
                "AvgMinimaxQHunterReward"
            ])
        
        if has_minimax_q_preys:
            self._active_metrics.extend([
                "MinimaxQPreys",
                "AvgMinimaxQPreyReward"
            ])
        
        #logger.info(f"Updated active metrics: {len(self._active_metrics)} metrics active based on current agents")
    def __len__(self) -> int:
        """Return the number of metrics."""
        return len(self._active_metrics)
    
    def __contains__(self, metric_name: str) -> bool:
        """Check if a metric is in the metrics list using 'in' operator."""
        return metric_name in self._active_metrics
    
    def __iter__(self):
        """Allow iteration over the metrics."""
        return iter(self._active_metrics)
    
    def __str__(self) -> str:
        """String representation of the ChartMetrics."""
        return f"ChartMetrics({len(self._active_metrics)} metrics: {', '.join(self._active_metrics)})"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"ChartMetrics(metrics={self._active_metrics})"


# Create a singleton instance for use throughout the application
chart_metrics = ChartMetrics()
