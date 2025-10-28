from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

# Backup
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    TripleBarrierConfig,
)
from hummingbot.strategy_v2.models.executors import TrackedOrder


class NeutralGridExecutorConfig(ExecutorConfigBase):
    type: Literal["neutral_grid_executor"] = "neutral_grid_executor"
    # Boundaries
    connector_name: str
    trading_pair: str
    start_price: Decimal
    end_price: Decimal
    side: TradeType = TradeType.BUY
    # Mode
    is_directional: bool = False
    # Profiling
    total_amount_quote: Decimal
    n_levels: int = 10  # Number of grid levels to create
    min_spread_between_orders: Optional[Decimal] = Decimal(
        "0.0005"
    )  # Optional fallback
    min_order_amount_quote: Decimal = Decimal("5")
    # Execution
    max_open_orders: int = 5
    max_orders_per_batch: Optional[int] = None
    order_frequency: int = 0
    activation_bounds: Optional[Decimal] = None
    safe_extra_spread: Decimal = Decimal("0.0001")
    # Trailing (window sliding) parameters
    grid_trailing_enabled: bool = False
    grid_trailing_cooldown_s: int = 1
    trailing_up_limit: Optional[Decimal] = None
    trailing_down_limit: Optional[Decimal] = None
    # Risk Management
    triple_barrier_config: TripleBarrierConfig
    leverage: int = 20
    level_id: Optional[str] = None
    deduct_base_fees: bool = False
    keep_position: bool = False
    coerce_tp_to_step: bool = True
    disable_first_level_tp: bool = (
        False  # Disable TP on first level for true grid behavior
    )


class GridLevelStates(Enum):
    NOT_ACTIVE = "NOT_ACTIVE"
    OPEN_ORDER_PLACED = "OPEN_ORDER_PLACED"
    OPEN_ORDER_FILLED = "OPEN_ORDER_FILLED"
    CLOSE_ORDER_PLACED = "CLOSE_ORDER_PLACED"
    COMPLETE = "COMPLETE"
    IDLE = "IDLE"


class GridLevel(BaseModel):
    id: str
    price: Decimal
    amount_quote: Decimal
    take_profit: Decimal
    side: TradeType
    open_order_type: OrderType
    take_profit_order_type: OrderType
    active_open_order: Optional[TrackedOrder] = None
    active_close_order: Optional[TrackedOrder] = None
    state: GridLevelStates = GridLevelStates.NOT_ACTIVE
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_state(self, logger=None):
        if self.state == GridLevelStates.IDLE:
            return
        if logger:
            logger().info(
                f"TrailingDeb: levelDeb {self.active_open_order.order_id if self.active_open_order else None} {self.state} {self.active_open_order.is_filled if self.active_open_order else None}="
            )
        if self.active_open_order is None:
            self.state = GridLevelStates.NOT_ACTIVE
        elif self.active_open_order.is_filled:
            self.state = GridLevelStates.OPEN_ORDER_FILLED
        else:
            self.state = GridLevelStates.OPEN_ORDER_PLACED
        if self.active_close_order is not None:
            if self.active_close_order.is_filled:
                self.state = GridLevelStates.COMPLETE
            else:
                self.state = GridLevelStates.CLOSE_ORDER_PLACED

    def reset_as_filled(self):
        self.state = GridLevelStates.OPEN_ORDER_FILLED

    def reset_open_order(self):
        self.active_open_order = None
        self.state = GridLevelStates.NOT_ACTIVE

    def reset_close_order(self):
        self.active_close_order = None
        self.state = GridLevelStates.OPEN_ORDER_FILLED

    def reset_level(self):
        self.active_open_order = None
        self.active_close_order = None
        self.state = GridLevelStates.NOT_ACTIVE
