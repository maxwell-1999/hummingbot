import asyncio
import logging
from decimal import Decimal, ROUND_DOWN
from pprint import pformat

import math
import time
from decimal import Decimal
from typing import Dict, List, Optional, Union

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import (
    OrderType,
    PositionAction,
    PriceType,
    TradeType,
)
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.core.data_type.order_candidate import (
    OrderCandidate,
    PerpetualOrderCandidate,
)
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.neutral_grid_executor.data_types import (
    GridLevel,
    GridLevelStates,
    NeutralGridExecutorConfig,
)
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import (
    CloseType,
    DummyTrackedFilledOrder,
    TrackedOrder,
)
from hummingbot.strategy_v2.utils.distributions import Distributions
# 2 types of levels:
# Short Level = Open:Sell@price, Close:Buy@l.price-spread (TP)
# Long Level = Open:Buy@price, Close:Sell@l.price+spread (TP)

# 3 types of grids:
# Long Grid : [start_price,end_price-1] long, idle
# Short Grid : idle, [start_price+1,end_price] short
# Neutral Grid : [start_price,CP-1] are long, 1idle, [CP+1,end_price] are short


class MDD:
    investment: Decimal = Decimal("0")
    worst_pnl: Decimal = Decimal("0")
    mdd: Decimal = Decimal("0")
    cdd: Decimal = Decimal("0")

    def __init__(self, investment: Decimal):
        self.investment = investment
        self.worst_pnl = Decimal("0")
        self.mdd = Decimal("0")
        self.cdd = Decimal("0")

    def update(self, pnl: Decimal, logger: HummingbotLogger):
        if pnl < self.worst_pnl:
            self.worst_pnl = pnl
        self.cdd = abs(pnl / self.investment)
        self.mdd = abs(self.worst_pnl / self.investment)
        return self.cdd

    def to_json(self):
        return {
            "investment": self.investment,
            "worst_pnl": self.worst_pnl,
            "mdd": (self.mdd * 100).quantize(Decimal("0.01")),
            "cdd": (self.cdd * 100).quantize(Decimal("0.01")),
        }


class NeutralGridExecutor(ExecutorBase):
    _logger = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger("g:")
        return cls._logger

    def log_level(self, level: GridLevel):
        self.logger().info(
            f"{level.side.name} Level@{level.price} {level.amount_quote} "
        )

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        config: NeutralGridExecutorConfig,
        update_interval: float = 1.0,
        max_retries: int = 10,
    ):
        """
        Initialize the PositionExecutor instance.

        :param strategy: The strategy to be used by the PositionExecutor.
        :param config: The configuration for the PositionExecutor, subclass of PositionExecutoConfig.
        :param update_interval: The interval at which the PositionExecutor should be updated, defaults to 1.0.
        :param max_retries: The maximum number of retries for the PositionExecutor, defaults to 5.
        """
        self.config: NeutralGridExecutorConfig = config
        if (
            config.triple_barrier_config.time_limit_order_type != OrderType.MARKET
            or config.triple_barrier_config.stop_loss_order_type != OrderType.MARKET
        ):
            error = "Only market orders are supported for time_limit and stop_loss"
            self.logger().error(error)
            raise ValueError(error)
        super().__init__(
            strategy=strategy,
            config=config,
            connectors=[config.connector_name],
            update_interval=update_interval,
        )
        self._trails = []
        self.open_order_price_type = (
            PriceType.BestBid if config.side == TradeType.BUY else PriceType.BestAsk
        )
        self.close_order_price_type = (
            PriceType.BestAsk if config.side == TradeType.BUY else PriceType.BestBid
        )
        self.close_order_side = (
            TradeType.BUY if config.side == TradeType.SELL else TradeType.SELL
        )
        self.trading_rules = self.get_trading_rules(
            self.config.connector_name, self.config.trading_pair
        )
        # Debug: Log trading rules to verify exchange min sizes and increments
        self.logger().info(
            f"TradingRules: pair={self.config.trading_pair} is_perp={self.is_perpetual} "
            f"min_notional={self.trading_rules.min_notional_size} "
            f"min_order_size={self.trading_rules.min_order_size} "
            f"min_base_inc={self.trading_rules.min_base_amount_increment} "
            f"min_price_inc={self.trading_rules.min_price_increment}"
        )
        # Initialize spacing before grid generation; will be set in _generate_grid_levels
        self.grid_placing_difference: Decimal = Decimal("0")
        # Initial (directional) seed order setup (simple/minimal) - must be before grid generation
        self._initial_order_level = None
        self._initial_order_placed = False
        self._base_amount_per_level = Decimal("0")
        self._quote_amount_per_level = Decimal("0")
        self._mdd = MDD(Decimal(self.config.total_amount_quote / self.config.leverage))
        self._initial_amount_base = Decimal("0")

        # Grid levels
        self.grid_levels = self._generate_grid_levels()
        self.logger().info(
            f"GenerateGridLevels: grid_levels {pformat([self.log_level(level) for level in self.grid_levels])}"
        )
        self.levels_by_state = {state: [] for state in GridLevelStates}
        self._close_order: Optional[TrackedOrder] = None
        self._filled_orders = []
        self._failed_orders = []
        self._canceled_orders = []

        self.step = Decimal("0")
        self.position_break_even_price = Decimal("0")
        self.position_size_base = Decimal("0")
        self.position_size_quote = Decimal("0")
        self.position_fees_quote = Decimal("0")
        self.position_pnl_quote = Decimal("0")
        self.position_pnl_pct = Decimal("0")
        self.open_liquidity_placed = Decimal("0")
        self.close_liquidity_placed = Decimal("0")
        self.realized_buy_size_quote = Decimal("0")
        self.realized_sell_size_quote = Decimal("0")
        self.realized_imbalance_quote = Decimal("0")
        self.realized_fees_quote = Decimal("0")
        self.realized_pnl_quote = Decimal("0")
        self.realized_pnl_pct = Decimal("0")
        self.max_open_creation_timestamp = 0
        self.max_close_creation_timestamp = 0
        self._open_fee_in_base = False

        self._trailing_stop_trigger_pct: Optional[Decimal] = None
        self._current_retries = 0
        self._max_retries = max_retries
        # Fixed arithmetic grid gap (distance) and slide cooldown controls
        self._slide_cooldown_s: int = getattr(config, "grid_trailing_cooldown_s", 30)
        self._last_slide_ts: float = 0.0

    @property
    def is_perpetual(self) -> bool:
        """
        Check if the exchange connector is perpetual.

        :return: True if the exchange connector is perpetual, False otherwise.
        """
        return self.is_perpetual_connector(self.config.connector_name)

    async def validate_sufficient_balance(self):
        mid_price = self.get_price(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        total_amount_base = self.config.total_amount_quote / mid_price
        if self.is_perpetual:
            order_candidate = PerpetualOrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=self.config.triple_barrier_config.open_order_type.is_limit_type(),
                order_type=self.config.triple_barrier_config.open_order_type,
                order_side=self.config.side,
                amount=total_amount_base,
                price=mid_price,
                leverage=Decimal(self.config.leverage),
            )
        else:
            order_candidate = OrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=self.config.triple_barrier_config.open_order_type.is_limit_type(),
                order_type=self.config.triple_barrier_config.open_order_type,
                order_side=self.config.side,
                amount=total_amount_base,
                price=mid_price,
            )
        adjusted_order_candidates = self.adjust_order_candidates(
            self.config.connector_name, [order_candidate]
        )
        if adjusted_order_candidates[0].amount == Decimal("0"):
            self.close_type = CloseType.INSUFFICIENT_BALANCE
            self.logger().error("Not enough budget to open position.")
            self.stop()

    def _generate_grid_levels(self):
        grid_levels = []
        price = self.get_price(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        # Get minimum notional and base amount increment from trading rules
        min_notional = max(
            self.config.min_order_amount_quote, self.trading_rules.min_notional_size
        )
        min_base_increment = self.trading_rules.min_base_amount_increment
        # Add safety margin to minimum notional to account for price movements and quantization
        min_notional_with_margin = min_notional * Decimal(
            "1.05"
        )  # 5% margin for safety
        # Calculate minimum base amount that satisfies both min_notional and quantization
        min_base_amount = max(
            min_notional_with_margin / price,  # Minimum from notional requirement
            min_base_increment
            * Decimal(
                str(math.ceil(float(min_notional) / float(min_base_increment * price)))
            ),
        )
        # Quantize the minimum base amount
        min_base_amount = (
            Decimal(str(math.ceil(float(min_base_amount) / float(min_base_increment))))
            * min_base_increment
        )
        # Verify the quantized amount meets minimum notional

        # Use n_levels directly from config
        n_levels = self.config.n_levels

        # Calculate quote amount per level
        base_amount_per_level = max(
            min_base_amount,
            Decimal(
                str(
                    math.floor(
                        float(self.config.total_amount_quote / (price * n_levels))
                        / float(min_base_increment)
                    )
                )
            )
            * min_base_increment,
        )
        quote_amount_per_level = base_amount_per_level * price

        # Adjust number of levels if total amount would be exceeded
        max_possible_levels = int(
            float(self.config.total_amount_quote) / float(quote_amount_per_level)
        )
        n_levels = min(n_levels, max_possible_levels)

        # Ensure we have at least one level
        n_levels = max(1, n_levels)

        # Calculate grid range and step size
        grid_range = (
            self.config.end_price - self.config.start_price
        ) / self.config.start_price
        self.logger().info(f"TrailingDeb: grid_range {n_levels}")
        # Generate price levels with even distribution (Decimal arithmetic)
        if n_levels > 1:
            self.grid_placing_difference = (
                self.config.end_price - self.config.start_price
            ) / Decimal(str(n_levels - 1))
            self.logger().info(
                f"TrailingDeb: distance {self.grid_placing_difference} {self.config.end_price - self.config.start_price} {n_levels - 1}"
            )
            prices = [
                self.config.start_price + self.grid_placing_difference * Decimal(str(i))
                for i in range(n_levels)
            ]
            self.step = grid_range / (n_levels - 1)
        else:
            # For single level, use mid-point of range
            mid_price = (self.config.start_price + self.config.end_price) / 2
            prices = [mid_price]
            self.step = grid_range
            self.grid_placing_difference = Decimal("0")

        # Create grid levels
        current_mid_price = self.get_price(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )

        for i, price in enumerate(prices):
            # For dynamic grid: disable TP since we use adjacent level logic
            level_take_profit = Decimal("0")  # Always disable TP for dynamic grid

            # Dynamic side determination based on current market price
            # Levels above mid_price = SELL, levels below = BUY
            dynamic_side = (
                TradeType.SELL if price > current_mid_price else TradeType.BUY
            )

            grid_levels.append(
                GridLevel(
                    id=f"L{i}",
                    price=price,
                    amount_quote=quote_amount_per_level,
                    take_profit=level_take_profit,  # Always 0 for dynamic grid
                    side=dynamic_side,  # Use dynamic side instead of config.side
                    open_order_type=self.config.triple_barrier_config.open_order_type,
                    take_profit_order_type=self.config.triple_barrier_config.take_profit_order_type,
                )
            )
        long_grid = not self.is_perpetual or (
            self.is_perpetual
            and self.config.is_directional
            and self.config.side == TradeType.BUY
        )
        short_grid = (
            self.is_perpetual
            and self.config.is_directional
            and self.config.side == TradeType.SELL
        )
        neutral_grid = self.is_perpetual and not self.config.is_directional
        self.logger().info(
            f"GenerateGridLevels: long_grid {long_grid} short_grid {short_grid} neutral_grid {neutral_grid}"
        )
        if long_grid:
            for level in grid_levels:
                level.side = TradeType.BUY
            grid_levels[len(grid_levels) - 1].state = GridLevelStates.IDLE
        elif short_grid:
            for level in grid_levels:
                level.side = TradeType.SELL
            grid_levels[0].state = GridLevelStates.IDLE
        # To make buys last as idle, or sells first as idle, whichever one is closer.

        else:
            # for neutral first levles till CP-1 are long, CP is idle, and from CP+1 to end_price are short
            first_sell_index = 0
            for i, level in enumerate(grid_levels):
                if level.price < current_mid_price:
                    level.side = TradeType.BUY
                    first_sell_index = i + 1
                else:
                    level.side = TradeType.SELL
            last_buy_index = first_sell_index - 1
            self.logger().info(
                f"GenerateGridLevels: first_sell_index {first_sell_index}"
            )
            self.logger().info(f"GenerateGridLevels: last_buy_index {last_buy_index}")
            # grid_levels[first_sell_index].state = GridLevelStates.IDLE
            last_buy_price_diff = current_mid_price - grid_levels[last_buy_index].price
            first_sell_price_diff = (
                grid_levels[first_sell_index].price - current_mid_price
            )
            self.logger().info(
                f"GenerateGridLevels: last_buy_price_diff {last_buy_price_diff} first_sell_price_diff {first_sell_price_diff}"
            )
            idle_index = first_sell_index
            if last_buy_price_diff < first_sell_price_diff:
                idle_index = last_buy_index
            grid_levels[idle_index].state = GridLevelStates.IDLE
        return grid_levels

    @property
    def end_time(self) -> Optional[float]:
        """
        Calculate the end time of the position based on the time limit

        :return: The end time of the position.
        """
        if not self.config.triple_barrier_config.time_limit:
            return None
        return self.config.timestamp + self.config.triple_barrier_config.time_limit

    @property
    def is_expired(self) -> bool:
        """
        Check if the position is expired.

        :return: True if the position is expired, False otherwise.
        """
        return self.end_time and self.end_time <= self._strategy.current_timestamp

    @property
    def is_trading(self):
        """
        Check if the position is trading.

        :return: True if the position is trading, False otherwise.
        """
        return (
            self.status == RunnableStatus.RUNNING
            and self.position_size_quote > Decimal("0")
        )

    @property
    def is_active(self):
        """
        Returns whether the executor is open or trading.
        """
        return self._status in [
            RunnableStatus.RUNNING,
            RunnableStatus.NOT_STARTED,
            RunnableStatus.SHUTTING_DOWN,
        ]

    def log_grid_levels(self):
        left_ext = self.grid_levels[0].price - self.grid_placing_difference
        right_ext = self.grid_levels[-1].price + self.grid_placing_difference
        price_list = ", ".join(
            [
                f"{level.side.name} @{str(level.price)} | {level.amount_quote} {level.state.value}"
                for level in self.grid_levels
            ]
        )

        self.logger().info(f" {left_ext} << {price_list} >> {right_ext}")

    async def control_task(self):
        """
        This method is responsible for controlling the task based on the status of the executor.

        :return: None
        """
        self.update_grid_levels()
        self.update_metrics()
        if self.status == RunnableStatus.RUNNING:
            if self.control_triple_barrier():
                self.cancel_open_orders()
                self._status = RunnableStatus.SHUTTING_DOWN
                return
            open_orders_to_create = self.get_open_orders_to_create()
            close_orders_to_create = self.get_close_orders_to_create()
            open_order_ids_to_cancel = self.get_open_order_ids_to_cancel()
            close_order_ids_to_cancel = self.get_close_order_ids_to_cancel()
            for level in open_orders_to_create:
                self.logger().info(
                    f"Creating open order for level {self.log_level(level)}"
                )
                self.adjust_and_place_open_order(level)
            for level in close_orders_to_create:
                self.logger().info(
                    f"Creating close order for level {self.log_level(level)}"
                )
                self.adjust_and_place_close_order(level)
            for orders_id_to_cancel in (
                open_order_ids_to_cancel + close_order_ids_to_cancel
            ):
                # TODO: Implement batch order cancel
                self._strategy.cancel(
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    order_id=orders_id_to_cancel,
                )
        elif self.status == RunnableStatus.SHUTTING_DOWN:
            await self.control_shutdown_process()
        self.evaluate_max_retries()

    def early_stop(self, keep_position: bool = False):
        """
        This method allows strategy to stop the executor early.

        :return: None
        """
        self.cancel_open_orders()
        self._status = RunnableStatus.SHUTTING_DOWN
        self.close_type = (
            CloseType.POSITION_HOLD if keep_position else CloseType.EARLY_STOP
        )

    def update_grid_levels(self, debug=False):
        self.levels_by_state = {state: [] for state in GridLevelStates}
        for level in self.grid_levels:
            level.update_state()
            self.levels_by_state[level.state].append(level)
        completed = self.levels_by_state[GridLevelStates.COMPLETE]
        # Get completed orders and store them in the filled orders list
        for level in completed:
            if (
                level.active_open_order.order.completely_filled_event.is_set()
                and level.active_close_order.order.completely_filled_event.is_set()
            ):
                self.levels_by_state[GridLevelStates.COMPLETE].remove(level)
                level.reset_level()
                self.levels_by_state[GridLevelStates.NOT_ACTIVE].append(level)

    async def control_shutdown_process(self):
        """
        Control the shutdown process of the executor, handling held positions separately
        """
        self.close_timestamp = self._strategy.current_timestamp
        open_orders_completed = self.open_liquidity_placed == Decimal("0")
        close_orders_completed = self.close_liquidity_placed == Decimal("0")

        if open_orders_completed and close_orders_completed:
            if self.close_type == CloseType.POSITION_HOLD:
                # Move filled orders to held positions instead of regular filled orders
                for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_FILLED]:
                    if level.active_open_order and level.active_open_order.order:
                        self._held_position_orders.append(
                            level.active_open_order.order.to_json()
                        )
                    level.reset_level()
                for level in self.levels_by_state[GridLevelStates.CLOSE_ORDER_PLACED]:
                    if level.active_close_order and level.active_close_order.order:
                        self._held_position_orders.append(
                            level.active_close_order.order.to_json()
                        )
                    level.reset_level()
                if len(self._held_position_orders) == 0:
                    self.close_type = CloseType.EARLY_STOP
                self.levels_by_state = {}
                self.stop()
            else:
                # Regular shutdown process for non-held positions
                order_execution_completed = self.position_size_base == Decimal("0")
                if order_execution_completed:
                    for level in self.levels_by_state[
                        GridLevelStates.OPEN_ORDER_FILLED
                    ]:
                        if level.active_open_order and level.active_open_order.order:
                            self._filled_orders.append(level.active_open_order.order)
                        level.reset_level()
                    for level in self.levels_by_state[
                        GridLevelStates.CLOSE_ORDER_PLACED
                    ]:
                        if level.active_close_order and level.active_close_order.order:
                            self._filled_orders.append(level.active_close_order.order)
                        level.reset_level()
                    if self._close_order and self._close_order.order:
                        self._filled_orders.append(self._close_order.order)
                        self._close_order = None
                    self.update_realized_pnl_metrics()
                    self.levels_by_state = {}
                    self.stop()
                else:
                    await self.control_close_order()
                    self._current_retries += 1
        else:
            self.cancel_open_orders()
        await self._sleep(5.0)

    async def control_close_order(self):
        """
        This method is responsible for controlling the close order. If the close order is filled and the open orders are
        completed, it stops the executor. If the close order is not placed, it places the close order. If the close order
        is not filled, it waits for the close order to be filled and requests the order information to the connector.
        """
        if self._close_order:
            in_flight_order = (
                self.get_in_flight_order(
                    self.config.connector_name, self._close_order.order_id
                )
                if not self._close_order.order
                else self._close_order.order
            )
            if in_flight_order:
                self._close_order.order = in_flight_order
            else:
                self._failed_orders.append(self._close_order.order_id)
                self._close_order = None
        elif not self.config.keep_position or self.close_type == CloseType.TAKE_PROFIT:
            self.place_close_order_and_cancel_open_orders(close_type=self.close_type)

    def adjust_and_place_open_order(self, level: GridLevel):
        """
        This method is responsible for adjusting the open order and placing it.

        :param level: The level to adjust and place the open order.
        :return: None
        """
        order_candidate = self._get_open_order_candidate(level)
        self.adjust_order_candidates(self.config.connector_name, [order_candidate])
        if order_candidate.amount > 0:
            # Stagger to avoid duplicate ms nonces on burst submissions
            # self._stagger_nonce_sync()
            order_id = self.place_order(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                order_type=self.config.triple_barrier_config.open_order_type,
                amount=order_candidate.amount,
                price=order_candidate.price,
                side=order_candidate.order_side,
                position_action=PositionAction.OPEN,
            )
            level.active_open_order = TrackedOrder(order_id=order_id)
            self.max_open_creation_timestamp = self._strategy.current_timestamp

    def adjust_and_place_close_order(self, level: GridLevel):
        order_candidate = self._get_close_order_candidate(level)
        self.adjust_order_candidates(self.config.connector_name, [order_candidate])
        if order_candidate.amount > 0:
            # Stagger to avoid duplicate ms nonces on burst submissions
            # self._stagger_nonce_sync()
            order_id = self.place_order(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                order_type=self.config.triple_barrier_config.take_profit_order_type,
                amount=order_candidate.amount,
                price=order_candidate.price,
                side=order_candidate.order_side,
                position_action=PositionAction.CLOSE,
            )
            level.active_close_order = TrackedOrder(order_id=order_id)
            self.logger().debug(
                f"Executor ID: {self.config.id} - Placing close order {order_id}"
            )

    def get_take_profit_price(self, level: GridLevel):
        if level.side == TradeType.BUY:
            return level.price + self.grid_placing_difference
        return level.price - self.grid_placing_difference

    def _get_open_order_candidate(self, level: GridLevel):
        entry_price = level.price
        if self.is_perpetual:
            return PerpetualOrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=self.config.triple_barrier_config.open_order_type.is_limit_type(),
                order_type=self.config.triple_barrier_config.open_order_type,
                order_side=level.side,  # Use level's dynamic side
                amount=level.amount_quote / self.mid_price,
                price=entry_price,
                leverage=Decimal(self.config.leverage),
            )
        return OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=self.config.triple_barrier_config.open_order_type.is_limit_type(),
            order_type=self.config.triple_barrier_config.open_order_type,
            order_side=level.side,  # Use level's dynamic side
            amount=level.amount_quote / self.mid_price,
            price=entry_price,
        )

    def _get_close_order_candidate(self, level: GridLevel):
        take_profit_price = self.get_take_profit_price(level)
        # Determine close order side - opposite of the level's side
        close_side = TradeType.SELL if level.side == TradeType.BUY else TradeType.BUY

        # For spot trading, fees are typically deducted from base asset when buying
        # So we need to account for fees when calculating close order amount
        amount = level.active_open_order.executed_amount_base
        if not self.is_perpetual and level.active_open_order:
            base_asset = self.config.trading_pair.split("-")[0]
            # If fees are paid in base asset, deduct them from the amount available for closing
            # This ensures the close order amount matches what's actually available after fees
            if level.active_open_order.fee_asset == base_asset and (
                self.config.deduct_base_fees or not self.is_perpetual
            ):
                amount = (
                    level.active_open_order.executed_amount_base
                    - level.active_open_order.cum_fees_base
                )
                # Ensure amount doesn't go negative
                amount = max(amount, Decimal("0"))

        if self.is_perpetual:
            return PerpetualOrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=self.config.triple_barrier_config.take_profit_order_type.is_limit_type(),
                order_type=self.config.triple_barrier_config.take_profit_order_type,
                order_side=close_side,  # Use opposite side for closing
                amount=amount,
                price=take_profit_price,
                leverage=Decimal(self.config.leverage),
            )
        return OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=self.config.triple_barrier_config.take_profit_order_type.is_limit_type(),
            order_type=self.config.triple_barrier_config.take_profit_order_type,
            order_side=close_side,  # Use opposite side for closing
            amount=amount,
            price=take_profit_price,
        )

    def update_metrics(self):
        self.mid_price = self.get_price(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
        )
        self.current_open_quote = self.get_price(
            self.config.connector_name,
            self.config.trading_pair,
            price_type=self.open_order_price_type,
        )
        self.current_close_quote = self.get_price(
            self.config.connector_name,
            self.config.trading_pair,
            price_type=self.close_order_price_type,
        )

        # Update grid sides based on current price for dynamic behavior
        # Sliding window: shift by one distance step if breached
        self._maybe_trailing_slide()

        self.update_all_pnl_metrics()

        # Log stats periodically (every 5 seconds)
        if hasattr(self, "_last_stats_log_time"):
            if self._strategy.current_timestamp - self._last_stats_log_time >= 20:
                self._log_trading_stats()
                self._last_stats_log_time = self._strategy.current_timestamp
        else:
            self._last_stats_log_time = self._strategy.current_timestamp
            self._log_trading_stats()

    def _log_trading_stats(self):
        """
        Log key trading statistics including volume_traded and other API metrics.
        This provides the same data that would be available via API without running the server.
        """
        self.log_grid_levels()
        try:
            # Calculate key metrics (same as API would show)
            volume_traded = float(self.filled_amount_quote)
            net_pnl = float(self.get_net_pnl_quote())
            net_pnl_pct = float(self.get_net_pnl_pct()) * 100  # Convert to percentage
            cum_fees = float(self.get_cum_fees_quote())

            # Grid level stats
            total_levels = len(self.grid_levels)
            active_orders = len(
                self.levels_by_state.get(GridLevelStates.OPEN_ORDER_PLACED, [])
            )
            filled_orders_count = len(self._filled_orders)

            # Position info
            position_size = float(self.position_size_quote)
            realized_pnl = float(self.realized_pnl_quote)
            unrealized_pnl = float(self.position_pnl_quote)
            break_even = float(self.position_break_even_price)

            # Debug: Show the calculation breakdown

            self.logger().info(
                f"├─ 💰 Volume Traded: Buy{self.realized_buy_size_quote:.4f} Sell{self.realized_sell_size_quote:.4f} {self.config.trading_pair.split('-')[1]}\n"
                f"├─ 📈 Net PnL: {self.realized_pnl_quote:.4f} ({net_pnl_pct:.2f}%)\n"
                f"├─ 💵 Realized PnL: {self.realized_pnl_quote:.4f}\n"
                f"├─ 📊 Unrealized PnL: {self.realized_pnl_quote:.4f}\n"
                f"├─ 💸 Total Fees: {self.realized_fees_quote:.4f}\n"
                f"├─ 🎯 Position Size: {self.position_size_quote:.4f}\n"
                f"├─ 📋 Filled Orders: {filled_orders_count}\n"
                f"├─ 💹 Mid Price: {float(self.mid_price):.4f}\n"
                f"├─ 💧 MDD: {self._mdd.to_json()['mdd']}%\n"
                f"├─ 💧 CDD: {self._mdd.to_json()['cdd']}%\n"
            )

        except Exception as e:
            self.logger().error(f"Error logging trading stats: {e}")

    def get_open_orders_to_create(self):
        """
        This method is responsible for controlling the open orders. Will check for each grid level if the order if there
        is an open order. If not, it will place a new orders from the proposed grid levels based on the current price,
        max open orders, max orders per batch, activation bounds and order frequency.
        For dynamic grid: only create orders that make sense based on current price vs level price.
        """

        # Filter levels by activation bounds and dynamic logic
        levels_allowed = self._filter_levels_by_activation_bounds()

        sorted_levels_by_proximity = self._sort_levels_by_proximity(levels_allowed)
        return sorted_levels_by_proximity[: self.config.max_orders_per_batch]

    def get_close_orders_to_create(self):
        """
        This method is responsible for controlling the take profit.
        For dynamic grid: TP is disabled - we use adjacent level logic instead.

        :return: Empty list - no automatic TP orders in dynamic grid
        """
        close_orders_proposal = []
        open_orders_filled = self.levels_by_state[GridLevelStates.OPEN_ORDER_FILLED]
        for level in open_orders_filled:
            close_orders_proposal.append(level)
        return close_orders_proposal

    def get_open_order_ids_to_cancel(self):
        if self.config.activation_bounds:
            open_orders_to_cancel = []
            open_orders_placed = [
                level.active_open_order
                for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_PLACED]
            ]
            for order in open_orders_placed:
                if not order:
                    continue
                price = order.price
                if price:
                    distance_pct = abs(price - self.mid_price) / self.mid_price
                    if distance_pct > self.config.activation_bounds:
                        open_orders_to_cancel.append(order.order_id)
                        self.logger().debug(
                            f"Executor ID: {self.config.id} - Canceling open order {order.order_id}"
                        )
            return open_orders_to_cancel
        return []

    def get_close_order_ids_to_cancel(self):
        """
        This method is responsible for controlling the close orders. It will check if the take profit is greater than the
        current price and cancel the close order.

        :return: None
        """
        if self.config.activation_bounds:
            close_orders_to_cancel = []
            close_orders_placed = [
                level.active_close_order
                for level in self.levels_by_state[GridLevelStates.CLOSE_ORDER_PLACED]
            ]
            for order in close_orders_placed:
                if not order:
                    continue
                price = order.price
                if price:
                    distance_to_mid = abs(price - self.mid_price) / self.mid_price
                    if distance_to_mid > self.config.activation_bounds:
                        close_orders_to_cancel.append(order.order_id)
            return close_orders_to_cancel
        return []

    def _filter_levels_by_activation_bounds(self):
        not_active_levels = self.levels_by_state[GridLevelStates.NOT_ACTIVE]
        return not_active_levels

    def _sort_levels_by_proximity(self, levels: List[GridLevel]):
        return sorted(levels, key=lambda level: abs(level.price - self.mid_price))

    def control_triple_barrier(self):
        """
        This method is responsible for controlling the barriers. It controls the stop loss, take profit, time limit and
        trailing stop.

        :return: None
        """
        sl = self.stop_loss_condition()
        # tp = self.take_profit_condition()
        ts = self.trailing_stop_condition()

        if sl:
            self.close_type = CloseType.STOP_LOSS
            return True
        elif self.is_expired:
            self.close_type = CloseType.TIME_LIMIT
            return True
        elif ts:
            self.close_type = CloseType.TRAILING_STOP
            return True
        # elif tp:
        #     self.close_type = CloseType.TAKE_PROFIT
        #     return True

        return False

    def take_profit_condition(self):
        """
        Take profit will be when the mid price is above the end price of the grid and there are no active executors.
        """
        if (
            self.mid_price > self.config.end_price
            if self.config.side == TradeType.BUY
            else self.mid_price < self.config.start_price
        ):
            return True
        return False

    def stop_loss_condition(self):
        """
        This method is responsible for controlling the stop loss. If the net pnl percentage is less than the stop loss
        percentage, it places the close order and cancels the open orders.

        :return: None
        """

        if self.config.triple_barrier_config.stop_loss:
            return self.position_pnl_pct <= -self.config.triple_barrier_config.stop_loss

        return False

    def trailing_stop_condition(self):
        if self.config.triple_barrier_config.trailing_stop:
            net_pnl_pct = self.position_pnl_pct
            if not self._trailing_stop_trigger_pct:
                if (
                    net_pnl_pct
                    > self.config.triple_barrier_config.trailing_stop.activation_price
                ):
                    self._trailing_stop_trigger_pct = (
                        net_pnl_pct
                        - self.config.triple_barrier_config.trailing_stop.trailing_delta
                    )
            else:
                if net_pnl_pct < self._trailing_stop_trigger_pct:
                    return True
                if (
                    net_pnl_pct
                    - self.config.triple_barrier_config.trailing_stop.trailing_delta
                    > self._trailing_stop_trigger_pct
                ):
                    self._trailing_stop_trigger_pct = (
                        net_pnl_pct
                        - self.config.triple_barrier_config.trailing_stop.trailing_delta
                    )
        return False

    def place_close_order_and_cancel_open_orders(
        self, close_type: CloseType, price: Decimal = Decimal("NaN")
    ):
        """
        This method is responsible for placing the close order and canceling the open orders. If the difference between
        the open filled amount and the close filled amount is greater than the minimum order size, it places the close
        order. It also cancels the open orders.

        :param close_type: The type of the close order.
        :param price: The price to be used in the close order.
        :return: None
        """
        self.cancel_open_orders()
        if self.position_size_base >= self.trading_rules.min_order_size:
            order_id = self.place_order(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                order_type=OrderType.MARKET,
                amount=self.position_size_base,
                price=price,
                side=self.close_order_side,
                position_action=PositionAction.CLOSE,
            )
            self._close_order = TrackedOrder(order_id=order_id)
            self.logger().debug(
                f"Executor ID: {self.config.id} - Placing close order {order_id}"
            )
        self.close_type = close_type
        self._status = RunnableStatus.SHUTTING_DOWN

    def cancel_open_orders(self):
        """
        This method is responsible for canceling the open orders.

        :return: None
        """
        open_order_placed = [
            level.active_open_order
            for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_PLACED]
        ]
        close_order_placed = [
            level.active_close_order
            for level in self.levels_by_state[GridLevelStates.CLOSE_ORDER_PLACED]
        ]
        for order in open_order_placed + close_order_placed:
            # TODO: Implement cancel batch orders
            if order:
                self._strategy.cancel(
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    order_id=order.order_id,
                )
                self.logger().debug("Removing open order")
                self.logger().debug(
                    f"Executor ID: {self.config.id} - Canceling open order {order.order_id}"
                )

    def get_custom_info(self) -> Dict:
        held_position_value = sum(
            [
                Decimal(order["executed_amount_quote"])
                for order in self._held_position_orders
            ]
        )

        return {
            "active_range": [
                (
                    self.grid_levels[0].price
                    if len(self.grid_levels) > 0
                    else Decimal("NaN")
                ),
                (
                    self.grid_levels[-1].price
                    if len(self.grid_levels) > 0
                    else Decimal("NaN")
                ),
            ],
            "mdd": self._mdd.to_json(),
            "trails": self._trails,
            "break_even_price": self.position_break_even_price,
            "position_size_base": self.position_size_base,
            "held_position_value": held_position_value,
            "failed_orders": self._failed_orders,
            "canceled_orders": self._canceled_orders,
            "realized_buy_size_quote": self.realized_buy_size_quote,
            "realized_sell_size_quote": self.realized_sell_size_quote,
            "realized_imbalance_quote": self.realized_imbalance_quote,
            "realized_fees_quote": self.realized_fees_quote,
            "realized_pnl_quote": self.realized_pnl_quote,
            "realized_pnl_pct": self.realized_pnl_pct,
            "position_size_quote": self.position_size_quote,
            "position_fees_quote": self.position_fees_quote,
            "position_pnl_quote": self.position_pnl_quote,
            "open_liquidity_placed": self.open_liquidity_placed,
            "close_liquidity_placed": self.close_liquidity_placed,
        }

    async def on_start(self):
        """
        This method is responsible for starting the executor and validating if the position is expired. The base method
        validates if there is enough balance to place the open order.

        :return: None
        """
        await super().on_start()
        self.update_metrics()
        tp = self.control_triple_barrier()

        if tp:
            self.logger().error(
                f"Grid is already expired by {self.close_type} {self.config.triple_barrier_config.stop_loss}."
            )

            self._status = RunnableStatus.SHUTTING_DOWN
        else:
            self.logger().info("InitOrder: attempting initial market order (if needed)")
            self._place_initial_market_order_if_needed()
            # tiny stagger to avoid duplicate nonce on connectors using ms nonce
            await self._sleep(0.01)

    def evaluate_max_retries(self):
        """
        This method is responsible for evaluating the maximum number of retries to place an order and stop the executor
        if the maximum number of retries is reached.

        :return: None
        """
        if self._current_retries > self._max_retries:
            self.close_type = CloseType.FAILED
            self.stop()

    def update_tracked_orders_with_order_id(self, order_id: str):
        """
        This method is responsible for updating the tracked orders with the information from the InFlightOrder, using
        the order_id as a reference.

        :param order_id: The order_id to be used as a reference.
        :return: None
        """
        self.update_grid_levels()
        in_flight_order = self.get_in_flight_order(self.config.connector_name, order_id)
        if in_flight_order:
            for level in self.grid_levels:
                if (
                    level.active_open_order
                    and level.active_open_order.order_id == order_id
                ):
                    level.active_open_order.order = in_flight_order
                if (
                    level.active_close_order
                    and level.active_close_order.order_id == order_id
                ):
                    level.active_close_order.order = in_flight_order
            if self._close_order and self._close_order.order_id == order_id:
                self._close_order.order = in_flight_order

    def process_order_created_event(
        self, _, market, event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]
    ):
        """
        This method is responsible for processing the order created event. Here we will update the TrackedOrder with the
        order_id.
        """
        self.update_tracked_orders_with_order_id(event.order_id)

    def process_order_filled_event(self, _, market, event: OrderFilledEvent):
        """
        This method is responsible for processing the order filled event. Here we will update the value of
        _total_executed_amount_backup, that can be used if the InFlightOrder
        is not available.
        """
        self.update_tracked_orders_with_order_id(event.order_id)

    def process_order_completed_event(
        self, _, market, event: Union[BuyOrderCompletedEvent, SellOrderCompletedEvent]
    ):
        """
        This method is responsible for processing the order completed event. Here we will check if the id is one of the
        tracked orders and update the state, and implement dynamic grid logic.
        """
        self.logger().info(
            f"Executor ID: {self.config.id} - Processing order completed event {event.order_id}"
        )
        self.update_tracked_orders_with_order_id(event.order_id)

        # Simple PNL tracking: Add completed order to filled_orders list
        self._add_completed_order_to_filled_orders(event.order_id)

    def process_order_canceled_event(
        self, _, market: ConnectorBase, event: OrderCancelledEvent
    ):
        """
        This method is responsible for processing the order canceled event
        """
        self.update_grid_levels()
        self.logger().info(
            f"Executor ID: {self.config.id} - Processing order canceled event {event.order_id}"
        )
        levels_open_order_placed = [
            level for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_PLACED]
        ]
        levels_close_order_placed = [
            level for level in self.levels_by_state[GridLevelStates.CLOSE_ORDER_PLACED]
        ]
        for level in levels_open_order_placed:
            if event.order_id == level.active_open_order.order_id:
                self._canceled_orders.append(level.active_open_order.order_id)
                self.max_open_creation_timestamp = 0
                level.reset_open_order()
        for level in levels_close_order_placed:
            if event.order_id == level.active_close_order.order_id:
                self._canceled_orders.append(level.active_close_order.order_id)
                self.max_close_creation_timestamp = 0
                level.reset_close_order()
        if self._close_order and event.order_id == self._close_order.order_id:
            self._canceled_orders.append(self._close_order.order_id)
            self._close_order = None

    def process_order_failed_event(self, _, market, event: MarketOrderFailureEvent):
        """
        This method is responsible for processing the order failed event. Here we will add the InFlightOrder to the
        failed orders list.
        """
        self.logger().info(
            f"Executor ID: {self.config.id} - Processing order failed event {event.order_id}"
        )
        self.update_grid_levels()
        levels_open_order_placed = [
            level for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_PLACED]
        ]
        levels_close_order_placed = [
            level for level in self.levels_by_state[GridLevelStates.CLOSE_ORDER_PLACED]
        ]
        for level in levels_open_order_placed:
            if event.order_id == level.active_open_order.order_id:
                self._failed_orders.append(level.active_open_order.order_id)
                self.max_open_creation_timestamp = 0
                level.reset_open_order()
        for level in levels_close_order_placed:
            if event.order_id == level.active_close_order.order_id:
                self._failed_orders.append(level.active_close_order.order_id)
                self.max_close_creation_timestamp = 0
                level.reset_close_order()
        if self._close_order and event.order_id == self._close_order.order_id:
            self._failed_orders.append(self._close_order.order_id)
            self._close_order = None

    def _add_completed_order_to_filled_orders(self, order_id: str):
        """
        Simple PNL tracking: Add completed order to filled_orders list.
        This is called when an order is completed to immediately account for it in PNL.
        """
        # Find the completed order in grid levels
        for level in self.grid_levels:
            # Check open orders
            if (
                level.active_open_order
                and level.active_open_order.order_id == order_id
                and level.active_open_order.order
                and level.active_open_order.is_filled
            ):
                if level.active_open_order.order not in self._filled_orders:
                    self._filled_orders.append(level.active_open_order.order)

                return

            # Check close orders
            if (
                level.active_close_order
                and level.active_close_order.order_id == order_id
                and level.active_close_order.order
                and level.active_close_order.is_filled
            ):
                if level.active_close_order.order not in self._filled_orders:
                    self._filled_orders.append(level.active_close_order.order)
                return

        # Check main close order
        if (
            self._close_order
            and self._close_order.order_id == order_id
            and self._close_order.order
            and self._close_order.is_filled
        ):
            order_json = self._close_order.order.to_json()
            if order_json not in self._filled_orders:
                self._filled_orders.append(order_json)

    def _maybe_trailing_slide(self):
        """
        Modified grid trailing: Add new grid levels and remove opposite edge levels.
        When price moves up beyond upper bound, add new level at top and remove bottom level.
        When price moves down beyond lower bound, add new level at bottom and remove top level.
        """

        # self.logger().info(
        #     f"TrailingDeb: self.distance {self.grid_placing_difference} "
        # )
        if self.grid_placing_difference <= Decimal("0"):
            return

        lower = self.config.start_price
        upper = self.config.end_price
        mid = self.mid_price
        gap = self.grid_placing_difference

        # Directional trailing enablement based on config limits
        trailing_up_limit_cfg = getattr(self.config, "trailing_up_limit", None)
        trailing_down_limit_cfg = getattr(self.config, "trailing_down_limit", None)
        allow_up = trailing_up_limit_cfg is not None and Decimal(
            str(trailing_up_limit_cfg)
        ) != Decimal("0")
        allow_down = trailing_down_limit_cfg is not None and Decimal(
            str(trailing_down_limit_cfg)
        ) != Decimal("0")

        # Check if price has moved beyond bounds
        if mid > upper:
            steps_up = (mid - upper) // self.grid_placing_difference

            if steps_up >= 1 and allow_up:
                self._slide_window_up()
        elif mid < lower:
            steps_down = (lower - mid) // self.grid_placing_difference

            if steps_down >= 1 and allow_down:
                self._slide_window_down()
        # else:
        #     self.logger().debug("TrailingDeb: Price within bounds, no trailing needed")

    def _slide_window_up(self):
        """
        Rotate-only upward slide:
        1) Cancel bottom-most open order (if any)
        2) Move that level to new top at old_max + gap
        3) Update start/end from current min/max
        4) Place order on the new top level
        """
        self.logger().info(
            "TrailingDeb: Starting upward rotate - removing bottom, adding top"
        )

        gap = self.grid_placing_difference
        if gap <= Decimal("0"):
            self.logger().info("TrailingDeb: gap is zero, skipping rotate-up")
            return

        # Identify bottom-most level and current max price via indices
        lowest_level = self.grid_levels[0]
        current_max_price = self.grid_levels[-1].price
        self.logger().info(
            f"TrailingDeb: Bottom level {lowest_level.id} at price {lowest_level.price} | current_max={current_max_price}"
        )

        # Enforce upward trailing limit (if configured and non-zero)
        trailing_up_limit_cfg = getattr(self.config, "trailing_up_limit", None)
        if trailing_up_limit_cfg is not None and Decimal(
            str(trailing_up_limit_cfg)
        ) != Decimal("0"):
            proposed_new_top = current_max_price + gap
            if proposed_new_top > Decimal(str(trailing_up_limit_cfg)):
                self.logger().info(
                    f"TrailingDeb: Upward rotate blocked - proposed_top {proposed_new_top} exceeds trailing_up_limit {trailing_up_limit_cfg}"
                )
                return

        # Cancel existing open order on the level we are rotating
        if lowest_level.active_open_order and lowest_level.active_open_order.order_id:
            self.logger().info(
                f"TrailingDeb: Canceling order {lowest_level.active_open_order.order_id} on bottom level {lowest_level.id}"
            )
            self._strategy.cancel(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                order_id=lowest_level.active_open_order.order_id,
            )
            lowest_level.reset_as_filled()
            lowest_level.active_open_order = DummyTrackedFilledOrder(
                order=lowest_level.active_open_order.order
            )

        # Remove the bottom-most level from index 0
        level_to_move = self.grid_levels.pop(0)

        # Move the level to the new top (append)
        new_top_price = current_max_price + gap
        level_to_move.reset_as_filled()
        level_to_move.price = new_top_price
        # Update its side based on current mid
        level_to_move.side = (
            TradeType.SELL if level_to_move.price > self.mid_price else TradeType.BUY
        )
        self.grid_levels.append(level_to_move)

        # Update window bounds from first/last elements
        self.config.start_price = self.grid_levels[0].price
        self.config.end_price = self.grid_levels[-1].price
        self.logger().info(
            f"TrailingDeb: New window bounds start={self.config.start_price}, end={self.config.end_price}"
        )

        # Place order on previous top (index n-2), leave new top intact
        if len(self.grid_levels) >= 2:
            prev_top_level = self.grid_levels[-2]
            prev_top_level.side = TradeType.BUY
            prev_top_level.reset_level()
            self._trails.append(
                {
                    "start_price": self.config.start_price,
                    "end_price": self.config.end_price,
                    "side": "up_trail",
                    "time": self._strategy.current_timestamp,
                }
            )
        else:
            self.logger().info(
                "TrailingDeb: Not enough levels to consider previous top (n < 2)"
            )

        self._last_slide_ts = self._strategy.current_timestamp
        self.logger().info("TrailingDeb: Upward rotate completed successfully")
        # Refresh level states to avoid stale references in cancellation logic
        self.update_grid_levels(debug=True)

    def _slide_window_down(self):
        """
        Rotate-only downward slide:
        1) Cancel top-most open order (if any)
        2) Move that level to new bottom at old_min - gap
        3) Update start/end from current min/max
        4) Place order on the new bottom level
        """
        self.logger().info(
            "TrailingDeb: Starting downward rotate - removing top, adding bottom"
        )

        gap = self.grid_placing_difference
        if gap <= Decimal("0"):
            self.logger().info("TrailingDeb: gap is zero, skipping rotate-down")
            return

        # Identify top-most level and current min price via indices
        highest_level = self.grid_levels[-1]
        current_min_price = self.grid_levels[0].price
        self.logger().info(
            f"TrailingDeb: Top level {highest_level.id} at price {highest_level.price} | current_min={current_min_price}"
        )

        # Enforce downward trailing limit (if configured and non-zero)
        trailing_down_limit_cfg = getattr(self.config, "trailing_down_limit", None)
        if trailing_down_limit_cfg is not None and Decimal(
            str(trailing_down_limit_cfg)
        ) != Decimal("0"):
            proposed_new_bottom = current_min_price - gap
            if proposed_new_bottom < Decimal(str(trailing_down_limit_cfg)):
                self.logger().info(
                    f"TrailingDeb: Downward rotate blocked - proposed_bottom {proposed_new_bottom} below trailing_down_limit {trailing_down_limit_cfg}"
                )
                return

        # Cancel existing open order on the level we are rotating
        if highest_level.active_open_order and highest_level.active_open_order.order_id:
            self.logger().info(
                f"TrailingDeb: Canceling order {highest_level.active_open_order.order_id} on top level {highest_level.id}"
            )
            self._strategy.cancel(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                order_id=highest_level.active_open_order.order_id,
            )
            highest_level.reset_as_filled()
            highest_level.active_open_order = DummyTrackedFilledOrder(
                order=highest_level.active_open_order.order
            )

        # Remove the top-most level from the end
        level_to_move = self.grid_levels.pop(-1)

        # Move the level to the new bottom (insert at 0)
        new_bottom_price = current_min_price - gap
        self.logger().info(
            f"TrailingDeb: Moving {level_to_move.id} -> new bottom price {new_bottom_price}"
        )
        level_to_move.reset_as_filled()
        level_to_move.price = new_bottom_price
        level_to_move.side = TradeType.BUY
        self.grid_levels.insert(0, level_to_move)

        # Update window bounds from first/last elements
        self.config.start_price = self.grid_levels[0].price
        self.config.end_price = self.grid_levels[-1].price

        # Place order on next level above new bottom (index 1), leave new bottom intact
        if len(self.grid_levels) >= 2:
            # next_low_level = self.grid_levels[1]
            # For lower trail, we defer order placement to the control loop instead of forcing it here
            next_low_level = self.grid_levels[1]
            next_low_level.side = TradeType.SELL
            next_low_level.reset_level()
            self._trails.append(
                {
                    "start_price": self.config.start_price,
                    "end_price": self.config.end_price,
                    "side": "down_trail",
                    "time": self._strategy.current_timestamp,
                }
            )
            self.logger().info(
                f"modified level: {next_low_level.price} {next_low_level.state.name}"
            )
            self.logger().info(
                f"replaced level: {level_to_move.price} {level_to_move.state.name}"
            )
        else:
            self.logger().info(
                "TrailingDeb: Not enough levels to consider neighbor above bottom (n < 2)"
            )

        self._last_slide_ts = self._strategy.current_timestamp
        self.logger().info("TrailingDeb: Downward rotate completed successfully")
        # Refresh level states to avoid stale references in cancellation logic
        self.update_grid_levels(debug=True)

    def update_all_pnl_metrics(self):
        """
        Professional PnL calculation with proper position tracking and average cost basis.

        Logic:
        1. Process orders chronologically to maintain running position and average cost
        2. Each sell realizes PnL against current average cost basis
        3. Unrealized PnL = remaining position marked to current market price
        4. All volume tracking and metrics work as expected
        """
        if len(self._filled_orders) == 0:
            self._reset_all_metrics()
            return
        # DEBUG: Log available keys for troubleshooting

        try:
            # Sort orders by actual fill timestamp for chronological processing
            def get_fill_timestamp(order):
                # Try to get the actual fill timestamp from order_fills
                order_fills = order.get("order_fills", {})
                if order_fills:
                    # Get the earliest fill timestamp (in case of multiple fills)
                    fill_timestamps = [
                        fill.get("fill_timestamp", 0) for fill in order_fills.values()
                    ]
                    if fill_timestamps:
                        return min(fill_timestamps)
                # Fallback to last_update_timestamp, then creation_timestamp
                return float(
                    order.get(
                        "last_update_timestamp", order.get("creation_timestamp", 0)
                    )
                )

            filled_jsons = [order.to_json() for order in self._filled_orders]

            sorted_orders = sorted(filled_jsons, key=get_fill_timestamp)
            # self.logger().info(f"PNLDeb: Order: {sorted_orders}")

            # Debug: Log order sequence with timestamps

            # Initialize tracking variables
            position_size_base = Decimal("0")  # Current position size
            position_cost_total = Decimal("0")  # Total cost of current position
            position_fees_total = Decimal("0")  # Total fees for current position

            realized_pnl_total = Decimal("0")  # Cumulative realized PnL
            realized_fees_total = Decimal("0")  # Cumulative realized fees

            # Volume tracking (for filled_amount_quote compatibility)
            total_buy_volume = Decimal("0")
            total_sell_volume = Decimal("0")
            total_fees = Decimal("0")

            buy_count = 0
            sell_count = 0

            # Process each order chronologically
            for i, order in enumerate(sorted_orders):
                try:
                    # Extract order data safely
                    # self.logger().info(f"PNLDeb: Order: {order}")
                    trade_type = order["trade_type"]
                    executed_base = Decimal(str(order["executed_amount_base"]))
                    executed_quote = Decimal(str(order["executed_amount_quote"]))
                    order_fees = Decimal(str(order.get("cumulative_fee_paid_quote", 0)))

                    # Calculate average execution price
                    if executed_base > 0:
                        avg_price = executed_quote / executed_base
                    else:
                        avg_price = Decimal(str(order.get("price", 0)))

                    total_fees += order_fees

                    # === SIMPLIFIED UNIFIED POSITION TRACKING ===
                    # Calculate position change (+ for BUY, - for SELL)
                    position_change = (
                        executed_base if trade_type == "BUY" else -executed_base
                    )
                    new_position_size = position_size_base + position_change

                    # Update volume counters
                    if trade_type == "BUY":
                        buy_count += 1
                        total_buy_volume += executed_quote
                    else:  # SELL
                        sell_count += 1
                        total_sell_volume += executed_quote

                    # === SIMPLE POSITION LOGIC ===
                    if trade_type == "BUY":
                        # BUY can: (a) add to long, (b) cover short partially/fully, (c) cover short and open long
                        prev_pos = position_size_base
                        prev_cost = position_cost_total

                        if prev_pos < 0:
                            # We are short; first cover up to the short size
                            short_qty = abs(prev_pos)
                            cover_amount = min(executed_base, short_qty)
                            fees_on_cover = (
                                order_fees * (cover_amount / executed_base)
                                if executed_base > 0
                                else Decimal("0")
                            )

                            if cover_amount > 0:
                                avg_short_price = prev_cost / short_qty
                                cover_pnl = (
                                    (avg_short_price * cover_amount)
                                    - (avg_price * cover_amount)
                                    - fees_on_cover
                                )
                                realized_pnl_total += cover_pnl
                                realized_fees_total += fees_on_cover

                                remaining_short = short_qty - cover_amount
                                if remaining_short > 0:
                                    # Still short after covering proportionally
                                    position_cost_total = (
                                        avg_short_price * remaining_short
                                    )
                                    position_fees_total = position_fees_total * (
                                        remaining_short / short_qty
                                    )
                                else:
                                    # Short fully covered
                                    position_cost_total = Decimal("0")
                                    position_fees_total = Decimal("0")

                            # Any remaining buy opens/adds to long
                            remaining_buy = executed_base - cover_amount
                            if remaining_buy > 0:
                                long_cost = avg_price * remaining_buy
                                # Start a new long with only the remaining amount (do not use full executed_quote)
                                position_cost_total = position_cost_total + long_cost
                                position_fees_total += order_fees - fees_on_cover
                        else:
                            # Add to existing/new long normally
                            position_cost_total = position_cost_total + executed_quote
                            position_fees_total += order_fees
                        # self.logger().info(
                        #     f"PNLDeb: Position cost total: {position_cost_total} position fees total: {position_fees_total}"
                        # )
                        # Finalize position size and average
                        position_size_base = new_position_size
                        if position_size_base > 0:
                            new_avg_cost = position_cost_total / position_size_base
                        elif position_size_base < 0:
                            new_avg_cost = position_cost_total / abs(position_size_base)
                        else:
                            new_avg_cost = Decimal("0")

                    else:  # SELL
                        # SELL can: (a) reduce long, (b) reduce long and open short, (c) add to short
                        prev_pos = position_size_base
                        prev_cost = position_cost_total

                        if prev_pos > 0:
                            long_qty = prev_pos
                            sell_from_long = min(executed_base, long_qty)
                            fees_on_long = (
                                order_fees * (sell_from_long / executed_base)
                                if executed_base > 0
                                else Decimal("0")
                            )

                            if sell_from_long > 0:
                                current_avg_cost = prev_cost / long_qty
                                sell_proceeds = avg_price * sell_from_long
                                cost_of_sold = current_avg_cost * sell_from_long
                                trade_realized_pnl = (
                                    sell_proceeds - cost_of_sold - fees_on_long
                                )
                                realized_pnl_total += trade_realized_pnl
                                realized_fees_total += fees_on_long

                                remaining_long = long_qty - sell_from_long
                                if remaining_long > 0:
                                    position_cost_total = (
                                        current_avg_cost * remaining_long
                                    )
                                    position_fees_total = position_fees_total * (
                                        remaining_long / long_qty
                                    )
                                else:
                                    position_cost_total = Decimal("0")
                                    position_fees_total = Decimal("0")

                            # If we sold more than we had, open short with remainder
                            remaining_sell = executed_base - sell_from_long
                            if remaining_sell > 0:
                                short_proceeds = avg_price * remaining_sell
                                position_cost_total = (
                                    position_cost_total + short_proceeds
                                )
                                position_fees_total += order_fees - fees_on_long
                        else:
                            # Opening/adding to short position
                            position_cost_total = position_cost_total + executed_quote
                            position_fees_total += order_fees

                        # Finalize avg for short side if short; for long it was handled above
                        if new_position_size < 0:
                            new_avg_cost = position_cost_total / abs(new_position_size)
                        elif new_position_size > 0:
                            new_avg_cost = (
                                position_cost_total / new_position_size
                                if new_position_size > 0
                                else Decimal("0")
                            )
                        else:
                            new_avg_cost = Decimal("0")

                    # Update position size
                    position_size_base = new_position_size

                except Exception as e:
                    self.logger().error(f"Error processing order {i + 1}: {e}")
                    continue

            # === FINAL CALCULATIONS ===

            # Set volume metrics (for filled_amount_quote compatibility)
            self.realized_buy_size_quote = total_buy_volume
            self.realized_sell_size_quote = total_sell_volume
            self.realized_fees_quote = total_fees
            self.realized_imbalance_quote = total_buy_volume - total_sell_volume

            # Set realized PnL
            self.realized_pnl_quote = realized_pnl_total
            self.realized_pnl_pct = (
                realized_pnl_total / total_buy_volume
                if total_buy_volume > 0
                else Decimal("0")
            )

            net_pnl = (
                self.realized_pnl_quote
                + self.position_pnl_quote
                + self.realized_fees_quote
            )
            self._mdd.update(net_pnl, self.logger())

            # Calculate unrealized PnL from remaining position
            if abs(position_size_base) > Decimal("0.000001"):  # Have remaining position
                if position_size_base > 0:
                    # Long position
                    current_avg_cost = (
                        position_cost_total / position_size_base
                        if position_size_base > 0
                        else Decimal("0")
                    )
                    current_market_value = position_size_base * self.mid_price
                    unrealized_pnl = current_market_value - position_cost_total

                else:
                    # Short position
                    current_market_value = abs(position_size_base) * self.mid_price
                    unrealized_pnl = (
                        position_cost_total - current_market_value
                    )  # Profit when market value < proceeds
                    current_avg_cost = (
                        position_cost_total / abs(position_size_base)
                        if position_size_base != 0
                        else Decimal("0")
                    )

                # Set position metrics
                self.position_size_base = abs(position_size_base)
                self.position_size_quote = abs(position_cost_total)
                self.position_break_even_price = current_avg_cost
                self.position_fees_quote = position_fees_total
                self.position_pnl_quote = unrealized_pnl
                self.position_pnl_pct = (
                    unrealized_pnl / abs(position_cost_total)
                    if abs(position_cost_total) > 0
                    else Decimal("0")
                )

            else:
                # No remaining position
                self.position_size_base = Decimal("0")
                self.position_size_quote = Decimal("0")
                self.position_break_even_price = Decimal("0")
                self.position_fees_quote = Decimal("0")
                self.position_pnl_quote = Decimal("0")
                self.position_pnl_pct = Decimal("0")

            # Update liquidity metrics (unchanged for compatibility)
            self.open_liquidity_placed = sum(
                level.active_open_order.executed_amount_quote
                for level in self.levels_by_state.get(
                    GridLevelStates.OPEN_ORDER_FILLED, []
                )
                if level.active_open_order and level.active_open_order.order
            )
            self.close_liquidity_placed = sum(
                level.active_close_order.executed_amount_quote
                for level in self.levels_by_state.get(
                    GridLevelStates.CLOSE_ORDER_PLACED, []
                )
                if level.active_close_order and level.active_close_order.order
            )

            # === FINAL SUMMARY ===

        except Exception as e:
            self.logger().error(f"Error updating all PnL metrics: {e}")
            self._reset_all_metrics()

    def _reset_all_metrics(self):
        """Reset all PnL and position metrics to zero"""
        # Realized metrics
        self.realized_buy_size_quote = Decimal("0")
        self.realized_sell_size_quote = Decimal("0")
        self.realized_imbalance_quote = Decimal("0")
        self.realized_fees_quote = Decimal("0")
        self.realized_pnl_quote = Decimal("0")
        self.realized_pnl_pct = Decimal("0")

        # Position metrics
        self.position_size_base = Decimal("0")
        self.position_size_quote = Decimal("0")
        self.position_fees_quote = Decimal("0")
        self.position_pnl_quote = Decimal("0")
        self.position_pnl_pct = Decimal("0")
        self.position_break_even_price = Decimal("0")

        # Liquidity metrics
        self.close_liquidity_placed = Decimal("0")
        self.open_liquidity_placed = Decimal("0")

    # Keep the old methods as thin wrappers for backward compatibility
    def update_position_metrics(self):
        """Backward compatibility wrapper - now just calls the combined function."""

    def update_realized_pnl_metrics(self):
        """Backward compatibility wrapper - now just calls the combined function."""

    def _reset_metrics(self):
        """Helper method to reset all PnL metrics - calls comprehensive reset"""
        self._reset_all_metrics()

    def get_net_pnl_quote(self) -> Decimal:
        """
        Calculate the net pnl in quote asset

        :return: The net pnl in quote asset.
        """
        return (
            self.position_pnl_quote + self.realized_pnl_quote
            if self.close_type != CloseType.POSITION_HOLD
            else self.realized_pnl_quote
        )

    def get_cum_fees_quote(self) -> Decimal:
        """
        Calculate the cumulative fees in quote asset

        :return: The cumulative fees in quote asset.
        """
        return (
            self.position_fees_quote + self.realized_fees_quote
            if self.close_type != CloseType.POSITION_HOLD
            else self.realized_fees_quote
        )

    @property
    def filled_amount_quote(self) -> Decimal:
        """
        Calculate the total trading volume in quote asset.
        This should represent actual trading volume, not position size.

        :return: The total trading volume in quote asset.
        """
        # Volume traded should be the sum of all executed orders (buy + sell volumes)
        # This represents actual trading activity, not position size
        matched_volume = self.realized_buy_size_quote + self.realized_sell_size_quote
        return matched_volume

    def get_net_pnl_pct(self) -> Decimal:
        """
        Calculate the net pnl percentage

        :return: The net pnl percentage.
        """
        return (
            self.get_net_pnl_quote() / self.filled_amount_quote
            if self.filled_amount_quote > 0
            else Decimal("0")
        )

    async def _sleep(self, delay: float):
        """
        This method is responsible for sleeping the executor for a specific time.

        :param delay: The time to sleep.
        :return: None
        """
        await asyncio.sleep(delay)

    def _place_initial_market_order_if_needed(self):
        try:
            self.logger().info("InitOrder: making initial order")

            if self._initial_order_placed or self._initial_order_level is None:
                return
            if not self.is_perpetual:
                self.logger().info("InitOrder: skipping (not a perpetual connector)")
                return
            # Use precomputed base amount to avoid drift from quote/mid changes
            amount_base = self._initial_amount_base
            if amount_base <= Decimal("0"):
                self.logger().info(
                    f"InitOrder: computed amount_base={amount_base}, skipping"
                )
                return
            # build MARKET candidate
            if self.is_perpetual:
                order_candidate = PerpetualOrderCandidate(
                    trading_pair=self.config.trading_pair,
                    is_maker=False,
                    order_type=OrderType.MARKET,
                    order_side=self._initial_order_level.side,
                    amount=amount_base,
                    price=self._initial_order_level.price,
                    leverage=Decimal(self.config.leverage),
                )
            else:
                order_candidate = OrderCandidate(
                    trading_pair=self.config.trading_pair,
                    is_maker=False,
                    order_type=OrderType.MARKET,
                    order_side=self._initial_order_level.side,
                    amount=amount_base,
                    price=self._initial_order_level.price,
                )
            self.adjust_order_candidates(self.config.connector_name, [order_candidate])
            if order_candidate.amount > 0:
                self.logger().info(
                    f"InitOrder: placing MARKET {order_candidate.order_side.name}, amount={order_candidate.amount} (base)"
                )
                # Stagger to avoid duplicate ms nonces on burst submissions
                self._stagger_nonce_sync()
                order_id = self.place_order(
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    order_type=OrderType.MARKET,
                    amount=order_candidate.amount,
                    price=Decimal("NaN"),
                    side=order_candidate.order_side,
                    position_action=PositionAction.OPEN,
                )
                self._initial_order_level.active_open_order = TrackedOrder(
                    order_id=order_id
                )
                self._initial_order_placed = True
                self.max_open_creation_timestamp = self._strategy.current_timestamp
            else:
                self.logger().info("InitOrder: adjusted amount is 0, skipping")
        except Exception as e:
            self.logger().error(f"InitOrder: failed to place market order: {e}")

    def _stagger_nonce_sync(self):
        """Ensure consecutive submissions occur in different milliseconds without touching connectors."""
        try:
            if not hasattr(self, "_last_submit_ms"):
                self._last_submit_ms = 0
            now_ms = int(time.time() * 1e3)
            if now_ms <= self._last_submit_ms:
                # sleep just enough to roll to next ms
                time.sleep(((self._last_submit_ms - now_ms) + 1) / 1000.0)
                now_ms = int(time.time() * 1e3)
            self._last_submit_ms = now_ms
        except Exception:
            pass
