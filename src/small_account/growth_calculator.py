"""
Growth Calculator for Small Accounts

Calculate growth projections, milestones, and time to financial goals.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class GrowthScenario(Enum):
    """Growth scenario types"""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class GrowthProjection:
    """Growth projection result"""

    scenario: GrowthScenario
    starting_balance: float
    monthly_return_rate: float
    final_balance: float
    total_return: float
    months_to_goal: int
    milestones: Dict[int, float]  # Month -> Balance


@dataclass
class Milestone:
    """Account milestone target"""

    target: float
    months: int
    probability: float
    strategy_required: str


class GrowthCalculator:
    """Calculate growth projections and milestones for small accounts"""

    # Scenario configurations
    SCENARIOS = {
        GrowthScenario.CONSERVATIVE: {
            "monthly_return": 0.08,  # 8% per month
            "volatility": 0.15,  # Low volatility
            "max_drawdown": 0.12,  # 12% max drawdown
            "description": "Steady growth with capital preservation",
        },
        GrowthScenario.MODERATE: {
            "monthly_return": 0.12,  # 12% per month
            "volatility": 0.25,  # Medium volatility
            "max_drawdown": 0.18,  # 18% max drawdown
            "description": "Balanced growth with moderate risk",
        },
        GrowthScenario.AGGRESSIVE: {
            "monthly_return": 0.15,  # 15% per month
            "volatility": 0.35,  # Higher volatility
            "max_drawdown": 0.25,  # 25% max drawdown
            "description": "High growth targeting rapid account building",
        },
        GrowthScenario.VERY_AGGRESSIVE: {
            "monthly_return": 0.20,  # 20% per month
            "volatility": 0.45,  # High volatility
            "max_drawdown": 0.35,  # 35% max drawdown
            "description": "Maximum growth with significant risk",
        },
    }

    # Common milestone targets
    MILESTONE_TARGETS = [1000, 2500, 5000, 10000, 25000, 50000, 100000]

    def __init__(self):
        pass

    def calculate_projections(
        self,
        starting_balance: float,
        months: int = 12,
        scenarios: Optional[List[GrowthScenario]] = None,
    ) -> Dict[GrowthScenario, GrowthProjection]:
        """Calculate growth projections for different scenarios"""

        if scenarios is None:
            scenarios = list(GrowthScenario)

        projections = {}

        for scenario in scenarios:
            config = self.SCENARIOS[scenario]
            monthly_rate = config["monthly_return"]

            # Calculate compound growth
            balance_history = [starting_balance]
            current_balance = starting_balance

            for month in range(months):
                # Add some realistic volatility
                volatility_factor = 1 + np.random.normal(0, config["volatility"] / 12)
                monthly_return = monthly_rate * volatility_factor

                current_balance *= 1 + monthly_return
                balance_history.append(current_balance)

            # Create milestone dictionary
            milestones = {}
            for i, balance in enumerate(balance_history):
                milestones[i] = balance

            final_balance = balance_history[-1]
            total_return = (final_balance - starting_balance) / starting_balance

            projections[scenario] = GrowthProjection(
                scenario=scenario,
                starting_balance=starting_balance,
                monthly_return_rate=monthly_rate,
                final_balance=final_balance,
                total_return=total_return,
                months_to_goal=months,
                milestones=milestones,
            )

        return projections

    def calculate_time_to_target(
        self, starting_balance: float, target_balance: float, monthly_return_rate: float
    ) -> int:
        """Calculate months needed to reach target balance"""

        if target_balance <= starting_balance:
            return 0

        if monthly_return_rate <= 0:
            return float("inf")

        # Using compound growth formula: A = P(1 + r)^n
        # Solving for n: n = log(A/P) / log(1 + r)
        months = np.log(target_balance / starting_balance) / np.log(
            1 + monthly_return_rate
        )

        return int(np.ceil(months))

    def get_milestone_analysis(
        self,
        starting_balance: float,
        scenario: GrowthScenario = GrowthScenario.MODERATE,
    ) -> List[Milestone]:
        """Get analysis of time to reach common milestones"""

        config = self.SCENARIOS[scenario]
        monthly_rate = config["monthly_return"]

        milestones = []

        for target in self.MILESTONE_TARGETS:
            if target > starting_balance:
                months = self.calculate_time_to_target(
                    starting_balance, target, monthly_rate
                )

                # Estimate probability based on scenario difficulty
                base_probability = {
                    GrowthScenario.CONSERVATIVE: 0.85,
                    GrowthScenario.MODERATE: 0.70,
                    GrowthScenario.AGGRESSIVE: 0.55,
                    GrowthScenario.VERY_AGGRESSIVE: 0.40,
                }

                # Reduce probability for longer timeframes
                time_factor = max(0.3, 1 - (months - 12) * 0.05)
                probability = base_probability[scenario] * time_factor

                # Recommend strategy based on target
                if target <= 2500:
                    strategy = "Momentum Scalping"
                elif target <= 10000:
                    strategy = "Swing Trading"
                elif target <= 25000:
                    strategy = "Breakout Trading"
                else:
                    strategy = "Mixed Strategies"

                milestones.append(
                    Milestone(
                        target=target,
                        months=months,
                        probability=min(probability, 1.0),
                        strategy_required=strategy,
                    )
                )

        return milestones

    def calculate_required_return(
        self, starting_balance: float, target_balance: float, months: int
    ) -> float:
        """Calculate required monthly return to reach target in given time"""

        if months <= 0 or target_balance <= starting_balance:
            return 0

        # Using compound growth formula: r = (A/P)^(1/n) - 1
        required_rate = (target_balance / starting_balance) ** (1 / months) - 1

        return required_rate

    def simulate_growth_path(
        self,
        starting_balance: float,
        monthly_return_rate: float,
        months: int,
        volatility: float = 0.2,
    ) -> pd.DataFrame:
        """Simulate a realistic growth path with volatility"""

        np.random.seed(42)  # For consistent demo results

        dates = pd.date_range(start=datetime.now(), periods=months + 1, freq="M")
        balances = [starting_balance]
        monthly_returns = [0]

        current_balance = starting_balance

        for month in range(months):
            # Add realistic volatility
            volatility_factor = 1 + np.random.normal(0, volatility)
            actual_return = monthly_return_rate * volatility_factor

            # Ensure no negative balance
            actual_return = max(actual_return, -0.5)

            current_balance *= 1 + actual_return
            balances.append(current_balance)
            monthly_returns.append(actual_return)

        return pd.DataFrame(
            {
                "Date": dates,
                "Balance": balances,
                "Monthly_Return": monthly_returns,
                "Cumulative_Return": [(b / starting_balance - 1) for b in balances],
            }
        )

    def get_scenario_comparison(
        self, starting_balance: float, months: int = 12
    ) -> pd.DataFrame:
        """Compare all scenarios side by side"""

        comparison_data = []

        for scenario in GrowthScenario:
            config = self.SCENARIOS[scenario]
            final_balance = starting_balance * (1 + config["monthly_return"]) ** months
            total_return = (final_balance - starting_balance) / starting_balance

            comparison_data.append(
                {
                    "Scenario": scenario.value.replace("_", " ").title(),
                    "Monthly Return": f"{config['monthly_return']:.1%}",
                    "Final Balance": f"${final_balance:,.0f}",
                    "Total Return": f"{total_return:.1%}",
                    "Max Drawdown": f"{config['max_drawdown']:.1%}",
                    "Risk Level": self._get_risk_level(scenario),
                    "Description": config["description"],
                }
            )

        return pd.DataFrame(comparison_data)

    def _get_risk_level(self, scenario: GrowthScenario) -> str:
        """Get risk level description for scenario"""
        risk_levels = {
            GrowthScenario.CONSERVATIVE: "🟢 Low",
            GrowthScenario.MODERATE: "🟡 Medium",
            GrowthScenario.AGGRESSIVE: "🟠 High",
            GrowthScenario.VERY_AGGRESSIVE: "🔴 Very High",
        }
        return risk_levels[scenario]
