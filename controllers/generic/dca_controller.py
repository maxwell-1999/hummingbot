from decimal import Decimal
from typing import List, Optional

from pydantic import Field

from hummingbot.core.data_type.common import (
    MarketDict,
    PositionMode,
    TradeType,
    PriceType,
)
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.dca_executor.data_types import (
    DCAExecutorConfig,
    DCAMode,
)
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
)
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class DCAControllerConfig(ControllerConfigBase):
    """
    Minimal configuration for a DCA controller that creates a single DCA executor.
    """

    controller_type: str = "generic"
    controller_name: str = "dca_controller"
    candles_config: List[CandlesConfig] = []

    # Account / connector
    connector_name: str
    trading_pair: str
    side: TradeType = TradeType.BUY
    position_mode: PositionMode = PositionMode.HEDGE
    leverage: int = 1

    # DCA parameters
    amounts_quote: List[Decimal] = Field(default_factory=list)
    prices: List[Decimal] = Field(default_factory=list)
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    time_limit: Optional[int] = None
    mode: DCAMode = DCAMode.MAKER
    activation_bounds: Optional[List[Decimal]] = None

    def update_markets(self, markets: MarketDict) -> MarketDict:
        # Register the single trading pair for this connector (runtime expects a string here)
        return markets.add_or_update(self.connector_name, self.trading_pair)  # type: ignore[arg-type]


class DCAController(ControllerBase):
    def __init__(self, config: DCAControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

    def active_executors(self) -> List[ExecutorInfo]:
        return [executor for executor in self.executors_info if executor.is_active]

    def determine_executor_actions(self) -> List[ExecutorAction]:
        # Emit a single DCA executor when none is active
        # Create price array
        if len(self.active_executors()) == 0 and len(self.config.amounts_quote) > 0:
            # Build prices dynamically if not provided or mismatched in length
            levels = len(self.config.amounts_quote)
            prices_to_use: List[Decimal]
            if len(self.config.prices) == levels and levels > 0:
                prices_to_use = self.config.prices
            else:
                mid_price = self.market_data_provider.get_price_by_type(
                    self.config.connector_name,
                    self.config.trading_pair,
                    PriceType.MidPrice,
                )
                print("mid_price", mid_price)
                step = Decimal("0.002")  # 2% step per level
                if self.config.side == TradeType.BUY:
                    mid_price = mid_price - 100
                    prices_to_use = [
                        mid_price * (Decimal("1") - step * Decimal(i + 1))
                        for i in range(levels)
                    ]
                else:
                    mid_price = mid_price + 100
                    prices_to_use = [
                        mid_price * (Decimal("1") + step * Decimal(i + 1))
                        for i in range(levels)
                    ]
            print("prices_to_use", prices_to_use)
            dca_config = DCAExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side=self.config.side,
                leverage=self.config.leverage,
                amounts_quote=self.config.amounts_quote,
                prices=prices_to_use,
                take_profit=self.config.take_profit,
                stop_loss=self.config.stop_loss,
                time_limit=self.config.time_limit,
                mode=self.config.mode,
                activation_bounds=self.config.activation_bounds,
                level_id=None,
            )
            return [
                CreateExecutorAction(
                    controller_id=self.config.id, executor_config=dca_config
                )
            ]
        return []

    async def update_processed_data(self):
        # No pre-processing required for this simple controller
        pass
