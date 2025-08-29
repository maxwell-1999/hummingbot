import asyncio
import logging
import math
from decimal import Decimal
from typing import Dict, List, Optional, Union

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import (
    OrderType,
    PositionAction,
    PriceType,
    TradeType,
)
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
from hummingbot.strategy_v2.executors.grid_executor.data_types import (
    GridExecutorConfig,
    GridLevel,
    GridLevelStates,
)
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder
from hummingbot.strategy_v2.utils.distributions import Distributions


class NeutralGridExecutor(ExecutorBase):
    _logger = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        strategy: ScriptStrategyBase,
        config: GridExecutorConfig,
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
        self.config: GridExecutorConfig = config
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
        # Grid levels
        self.grid_levels = self._generate_grid_levels()
        self.levels_by_state = {state: [] for state in GridLevelStates}
        self._close_order: Optional[TrackedOrder] = None
        self._filled_orders = []
        self._failed_orders = []
        self._canceled_orders = []
        self.logger().info(
            f"GridExecutor: Generating dynamic grid levels (TP disabled - using adjacent level logic)"
        )
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
        min_quote_amount = min_base_amount * price

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

        # Generate price levels with even distribution
        if n_levels > 1:
            prices = Distributions.linear(
                n_levels, float(self.config.start_price), float(self.config.end_price)
            )
            self.step = grid_range / (n_levels - 1)
        else:
            # For single level, use mid-point of range
            mid_price = (self.config.start_price + self.config.end_price) / 2
            prices = [mid_price]
            self.step = grid_range

        # Create grid levels
        for i, price in enumerate(prices):
            # For dynamic grid: disable TP since we use adjacent level logic
            level_take_profit = Decimal("0")  # Always disable TP for dynamic grid

            # Dynamic side determination based on current market price
            # Levels above mid_price = SELL, levels below = BUY
            current_mid_price = self.get_price(
                self.config.connector_name, self.config.trading_pair, PriceType.MidPrice
            )
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
        # Log grid creation details
        self.logger().info(
            f"""GridExecutor: Dynamic Grid Created
            ---------------
            Total levels: {len(grid_levels)} (requested: {self.config.n_levels})
            Price range: {self.config.start_price} - {self.config.end_price}
            Current mid price: {current_mid_price}
            Amount per level: {quote_amount_per_level}
            Grid levels: {[f"{g.id}: {g.price} ({g.side.name})" for g in grid_levels]}
            Note: Sides are dynamic - SELL above current price, BUY below current price
            """
        )
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
                self.adjust_and_place_open_order(level)
            for level in close_orders_to_create:
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

    def update_grid_levels(self):
        self.levels_by_state = {state: [] for state in GridLevelStates}
        for level in self.grid_levels:
            level.update_state()
            self.levels_by_state[level.state].append(level)

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
                            self._filled_orders.append(
                                level.active_open_order.order.to_json()
                            )
                        level.reset_level()
                    for level in self.levels_by_state[
                        GridLevelStates.CLOSE_ORDER_PLACED
                    ]:
                        if level.active_close_order and level.active_close_order.order:
                            self._filled_orders.append(
                                level.active_close_order.order.to_json()
                            )
                        level.reset_level()
                    if self._close_order and self._close_order.order:
                        self._filled_orders.append(self._close_order.order.to_json())
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
        return (
            level.price * (1 + level.take_profit)
            if self.config.side == TradeType.BUY
            else level.price * (1 - level.take_profit)
        )

    def _get_open_order_candidate(self, level: GridLevel):
        if (level.side == TradeType.BUY and level.price >= self.current_open_quote) or (
            level.side == TradeType.SELL and level.price <= self.current_open_quote
        ):
            entry_price = (
                self.current_open_quote * (1 - self.config.safe_extra_spread)
                if level.side == TradeType.BUY
                else self.current_open_quote * (1 + self.config.safe_extra_spread)
            )
        else:
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

        if (
            level.side == TradeType.BUY
            and take_profit_price <= self.current_close_quote
        ) or (
            level.side == TradeType.SELL
            and take_profit_price >= self.current_close_quote
        ):
            take_profit_price = (
                self.current_close_quote * (1 + self.config.safe_extra_spread)
                if level.side == TradeType.BUY
                else self.current_close_quote * (1 - self.config.safe_extra_spread)
            )
        if (
            level.active_open_order.fee_asset == self.config.trading_pair.split("-")[0]
            and self.config.deduct_base_fees
        ):
            amount = (
                level.active_open_order.executed_amount_base
                - level.active_open_order.cum_fees_base
            )
            self._open_fee_in_base = True
        else:
            amount = level.active_open_order.executed_amount_base
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
        self._update_grid_sides_based_on_price()

        self.update_all_pnl_metrics()

        # Log stats periodically (every 30 seconds)
        if hasattr(self, "_last_stats_log_time"):
            if self._strategy.current_timestamp - self._last_stats_log_time >= 30:
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
        try:
            self.logger().info("=== COMPREHENSIVE TRADING STATS ===")

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
                f"PnL Breakdown: Realized={realized_pnl:.4f} + Unrealized={unrealized_pnl:.4f} = Net={net_pnl:.4f}"
            )

            self.logger().info(
                f"\n📊 TRADING STATS - {self.config.id}\n"
                f"├─ 💰 Volume Traded: {volume_traded:.4f} {self.config.trading_pair.split('-')[1]}\n"
                f"├─ 📈 Net PnL: {net_pnl:.4f} ({net_pnl_pct:.2f}%)\n"
                f"├─ 💵 Realized PnL: {realized_pnl:.4f}\n"
                f"├─ 📊 Unrealized PnL: {unrealized_pnl:.4f}\n"
                f"├─ 💸 Total Fees: {cum_fees:.4f}\n"
                f"├─ 🎯 Position Size: {position_size:.4f}\n"
                f"├─ 🏆 Break-Even Price: {break_even:.4f}\n"
                f"├─ 📋 Filled Orders: {filled_orders_count}\n"
                f"├─ 🔄 Active Orders: {active_orders}/{total_levels}\n"
                f"├─ 💹 Mid Price: {float(self.mid_price):.4f}\n"
                f"└─ ⚡ Buy Vol: {float(self.realized_buy_size_quote):.4f} | Sell Vol: {float(self.realized_sell_size_quote):.4f}"
            )

        except Exception as e:
            self.logger().error(f"Error logging trading stats: {e}")

    def get_trading_stats_dict(self) -> Dict:
        """
        Get comprehensive trading statistics as a dictionary.
        This returns the same data that would be available via API.
        """
        try:
            return {
                "executor_id": self.config.id,
                "trading_pair": self.config.trading_pair,
                "mid_price": float(self.mid_price),
                # Volume metrics (API equivalent)
                "volume_traded": float(self.filled_amount_quote),
                "realized_buy_volume": float(self.realized_buy_size_quote),
                "realized_sell_volume": float(self.realized_sell_size_quote),
                # PnL metrics (API equivalent)
                "net_pnl_quote": float(self.get_net_pnl_quote()),
                "net_pnl_pct": float(self.get_net_pnl_pct()) * 100,
                "realized_pnl_quote": float(self.realized_pnl_quote),
                "unrealized_pnl_quote": float(self.position_pnl_quote),
                # Fee metrics
                "cumulative_fees_quote": float(self.get_cum_fees_quote()),
                "realized_fees_quote": float(self.realized_fees_quote),
                "position_fees_quote": float(self.position_fees_quote),
                # Position metrics
                "position_size_quote": float(self.position_size_quote),
                "position_size_base": float(self.position_size_base),
                "break_even_price": float(self.position_break_even_price),
                # Order metrics
                "filled_orders_count": len(self._filled_orders),
                "failed_orders_count": len(self._failed_orders),
                "canceled_orders_count": len(self._canceled_orders),
                # Grid metrics
                "total_levels": len(self.grid_levels),
                "active_orders": len(
                    self.levels_by_state.get(GridLevelStates.OPEN_ORDER_PLACED, [])
                ),
                "open_order_filled": len(
                    self.levels_by_state.get(GridLevelStates.OPEN_ORDER_FILLED, [])
                ),
                "close_order_placed": len(
                    self.levels_by_state.get(GridLevelStates.CLOSE_ORDER_PLACED, [])
                ),
                "complete_levels": len(
                    self.levels_by_state.get(GridLevelStates.COMPLETE, [])
                ),
                # Liquidity metrics
                "open_liquidity_placed": float(self.open_liquidity_placed),
                "close_liquidity_placed": float(self.close_liquidity_placed),
                # Status
                "status": self.status.name,
                "is_trading": self.is_trading,
                "timestamp": self._strategy.current_timestamp,
            }
        except Exception as e:
            self.logger().error(f"Error getting trading stats: {e}")
            return {}

    def print_trading_stats(self):
        """
        Print comprehensive trading statistics to console.
        Call this method manually to get instant stats display.
        """
        stats = self.get_trading_stats_dict()
        if not stats:
            return

        print(f"\n{'=' * 60}")
        print(f"🚀 NEUTRAL GRID EXECUTOR STATS - {stats['executor_id']}")
        print(f"{'=' * 60}")
        print(f"📊 Trading Pair: {stats['trading_pair']}")
        print(f"💹 Mid Price: {stats['mid_price']:.4f}")
        print(f"⚡ Status: {stats['status']} | Trading: {stats['is_trading']}")
        print(f"\n📈 VOLUME & PNL METRICS:")
        print(f"├─ 💰 Volume Traded: {stats['volume_traded']:.4f}")
        print(
            f"├─ 📈 Net PnL: {stats['net_pnl_quote']:.4f} ({stats['net_pnl_pct']:.2f}%)"
        )
        print(f"├─ 💵 Realized PnL: {stats['realized_pnl_quote']:.4f}")
        print(f"├─ 📊 Unrealized PnL: {stats['unrealized_pnl_quote']:.4f}")
        print(f"└─ 💸 Total Fees: {stats['cumulative_fees_quote']:.4f}")
        print(f"\n🎯 POSITION METRICS:")
        print(f"├─ 📦 Position Size: {stats['position_size_quote']:.4f} quote")
        print(f"├─ 🎯 Break Even: {stats['break_even_price']:.4f}")
        print(f"├─ ⬆️ Buy Volume: {stats['realized_buy_volume']:.4f}")
        print(f"└─ ⬇️ Sell Volume: {stats['realized_sell_volume']:.4f}")
        print(f"\n📋 ORDER METRICS:")
        print(f"├─ ✅ Filled Orders: {stats['filled_orders_count']}")
        print(f"├─ ❌ Failed Orders: {stats['failed_orders_count']}")
        print(f"├─ 🚫 Canceled Orders: {stats['canceled_orders_count']}")
        print(f"└─ 🔄 Active Orders: {stats['active_orders']}/{stats['total_levels']}")
        print(f"\n🏗️ GRID METRICS:")
        print(f"├─ 🔵 Open Filled: {stats['open_order_filled']}")
        print(f"├─ 🔴 Close Placed: {stats['close_order_placed']}")
        print(f"├─ ✅ Complete: {stats['complete_levels']}")
        print(f"├─ 💧 Open Liquidity: {stats['open_liquidity_placed']:.4f}")
        print(f"└─ 💧 Close Liquidity: {stats['close_liquidity_placed']:.4f}")
        print(f"{'=' * 60}\n")

    def get_open_orders_to_create(self):
        """
        This method is responsible for controlling the open orders. Will check for each grid level if the order if there
        is an open order. If not, it will place a new orders from the proposed grid levels based on the current price,
        max open orders, max orders per batch, activation bounds and order frequency.
        For dynamic grid: only create orders that make sense based on current price vs level price.
        """
        n_open_orders = len(
            [
                level.active_open_order
                for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_PLACED]
            ]
        )
        if (
            self.max_open_creation_timestamp
            > self._strategy.current_timestamp - self.config.order_frequency
            or n_open_orders >= self.config.max_open_orders
        ):
            return []

        # Filter levels by activation bounds and dynamic logic
        levels_allowed = self._filter_levels_by_activation_bounds()

        # Additional filtering for dynamic grid logic
        dynamic_levels_allowed = []
        current_price = self.mid_price

        for level in levels_allowed:
            # Only allow levels where the side makes sense relative to current price
            if (level.side == TradeType.SELL and level.price > current_price) or (
                level.side == TradeType.BUY and level.price <= current_price
            ):
                dynamic_levels_allowed.append(level)

        sorted_levels_by_proximity = self._sort_levels_by_proximity(
            dynamic_levels_allowed
        )
        return sorted_levels_by_proximity[: self.config.max_orders_per_batch]

    def get_close_orders_to_create(self):
        """
        This method is responsible for controlling the take profit.
        For dynamic grid: TP is disabled - we use adjacent level logic instead.

        :return: Empty list - no automatic TP orders in dynamic grid
        """
        # Dynamic grid doesn't use automatic TP - we place orders on adjacent levels instead
        return []

    def get_open_order_ids_to_cancel(self):
        if self.config.activation_bounds:
            open_orders_to_cancel = []
            open_orders_placed = [
                level.active_open_order
                for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_PLACED]
            ]
            for order in open_orders_placed:
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
                price = order.price
                if price:
                    distance_to_mid = abs(price - self.mid_price) / self.mid_price
                    if distance_to_mid > self.config.activation_bounds:
                        close_orders_to_cancel.append(order.order_id)
            return close_orders_to_cancel
        return []

    def _filter_levels_by_activation_bounds(self):
        not_active_levels = self.levels_by_state[GridLevelStates.NOT_ACTIVE]
        if self.config.activation_bounds:
            # For dynamic grid, filter based on each level's individual side
            filtered_levels = []
            for level in not_active_levels:
                if level.side == TradeType.BUY:
                    activation_bounds_price = self.mid_price * (
                        1 - self.config.activation_bounds
                    )
                    if level.price >= activation_bounds_price:
                        filtered_levels.append(level)
                else:  # level.side == TradeType.SELL
                    activation_bounds_price = self.mid_price * (
                        1 + self.config.activation_bounds
                    )
                    if level.price <= activation_bounds_price:
                        filtered_levels.append(level)
            return filtered_levels
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
            "levels_by_state": {
                key.name: value for key, value in self.levels_by_state.items()
            },
            "filled_orders": self._filled_orders,
            "held_position_orders": self._held_position_orders,
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
            "break_even_price": self.position_break_even_price,
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
        self.update_tracked_orders_with_order_id(event.order_id)

        # Simple PNL tracking: Add completed order to filled_orders list
        self._add_completed_order_to_filled_orders(event.order_id)

        # Dynamic grid logic: when an order fills, place opposite orders on adjacent levels
        self._handle_dynamic_grid_fill(event.order_id)

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
                order_json = level.active_open_order.order.to_json()
                if order_json not in self._filled_orders:
                    self._filled_orders.append(order_json)
                    self.logger().debug(
                        f"Added completed order {order_id} to PNL tracking"
                    )
                return

            # Check close orders
            if (
                level.active_close_order
                and level.active_close_order.order_id == order_id
                and level.active_close_order.order
                and level.active_close_order.is_filled
            ):
                order_json = level.active_close_order.order.to_json()
                if order_json not in self._filled_orders:
                    self._filled_orders.append(order_json)
                    self.logger().debug(
                        f"Added completed order {order_id} to PNL tracking"
                    )
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
                self.logger().debug(
                    f"Added completed close order {order_id} to PNL tracking"
                )

    def _handle_dynamic_grid_fill(self, filled_order_id: str):
        """
        Handle dynamic grid logic when an order is filled.
        Check i+1 and i-1 levels and place opposite orders if they're empty.
        """
        self.logger().debug(
            f"AutoFillDeb: Handling dynamic grid fill for order: {filled_order_id}"
        )

        # Find which level was filled
        filled_level_index = None
        filled_level = None

        for i, level in enumerate(self.grid_levels):
            if (
                level.active_open_order
                and level.active_open_order.order_id == filled_order_id
            ):
                self.logger().debug(
                    f"AutoFillDeb: Found order {filled_order_id} at level {i} ({level.id}), "
                    f"is_filled: {level.active_open_order.is_filled}, "
                    f"state: {level.state}"
                )
                # Check if order is filled (either fully or partially for our purposes)
                if (
                    level.active_open_order.is_filled
                    or level.state == GridLevelStates.OPEN_ORDER_FILLED
                ):
                    filled_level_index = i
                    filled_level = level
                    break

        if filled_level_index is None or filled_level is None:
            self.logger().debug(
                f"AutoFillDeb: Order {filled_order_id} not found in any grid level or not filled"
            )
            return

        self.logger().info(
            f"AutoFillDeb: Order filled at level {filled_level_index} ({filled_level.id}), "
            f"checking adjacent levels for placement"
        )

        # Check adjacent levels and place opposite orders
        self._place_adjacent_orders(filled_level_index, filled_level)

    def _place_adjacent_orders(self, filled_index: int, filled_level: GridLevel):
        """
        Place opposite orders on adjacent levels based on dynamic grid logic.
        Handles all states: NOT_ACTIVE, OPEN_ORDER_FILLED (needs remake), etc.
        """
        current_price = self.mid_price
        self.logger().debug(
            f"AutoFillDeb: Checking adjacent levels for filled_index: {filled_index}, current_price: {current_price}"
        )

        # Check level above (i+1) for SELL order
        if filled_index + 1 < len(self.grid_levels):
            upper_level = self.grid_levels[filled_index + 1]
            self.logger().debug(
                f"AutoFillDeb: Upper level {filled_index + 1} ({upper_level.id}): "
                f"price={upper_level.price}, state={upper_level.state}, "
                f"side={upper_level.side}, has_order={upper_level.active_open_order is not None}"
            )

            # Case 1: Level is completely vacant (NOT_ACTIVE)
            if (
                upper_level.price > current_price
                and upper_level.state == GridLevelStates.NOT_ACTIVE
            ):
                upper_level.side = TradeType.SELL
                self.logger().info(
                    f"AutoFillDeb: Placing SELL order on vacant upper level {upper_level.id}"
                )
                self._place_dynamic_order(upper_level)

            # Case 2: Level has filled open order (OPEN_ORDER_FILLED) - reset and remake
            elif (
                upper_level.state == GridLevelStates.OPEN_ORDER_FILLED
                and upper_level.price > current_price
            ):
                # For dynamic grid: reset filled levels since we don't use TP
                self.logger().info(
                    f"AutoFillDeb: Resetting filled upper level {upper_level.id} to place new SELL order"
                )
                upper_level.reset_level()  # Reset to NOT_ACTIVE
                upper_level.side = TradeType.SELL
                self._place_dynamic_order(upper_level)
            else:
                self.logger().debug(
                    f"AutoFillDeb: Upper level {upper_level.id} not eligible: "
                    f"price_check={upper_level.price > current_price}, "
                    f"state_check={upper_level.state == GridLevelStates.OPEN_ORDER_FILLED}"
                )

        # Check level below (i-1) for BUY order
        if filled_index - 1 >= 0:
            lower_level = self.grid_levels[filled_index - 1]
            self.logger().debug(
                f"AutoFillDeb: Lower level {filled_index - 1} ({lower_level.id}): "
                f"price={lower_level.price}, state={lower_level.state}, "
                f"side={lower_level.side}, has_order={lower_level.active_open_order is not None}"
            )

            # Case 1: Level is completely vacant (NOT_ACTIVE)
            if (
                lower_level.price <= current_price
                and lower_level.state == GridLevelStates.NOT_ACTIVE
            ):
                lower_level.side = TradeType.BUY
                self.logger().info(
                    f"AutoFillDeb: Placing BUY order on vacant lower level {lower_level.id}"
                )
                self._place_dynamic_order(lower_level)

            # Case 2: Level has filled open order (OPEN_ORDER_FILLED) - reset and remake
            elif (
                lower_level.state == GridLevelStates.OPEN_ORDER_FILLED
                and lower_level.price <= current_price
            ):
                # For dynamic grid: reset filled levels since we don't use TP
                self.logger().info(
                    f"AutoFillDeb: Resetting filled lower level {lower_level.id} to place new BUY order"
                )
                lower_level.reset_level()  # Reset to NOT_ACTIVE
                lower_level.side = TradeType.BUY
                self._place_dynamic_order(lower_level)
            else:
                self.logger().debug(
                    f"AutoFillDeb: Lower level {lower_level.id} not eligible: "
                    f"price_check={lower_level.price <= current_price}, "
                    f"state_check={lower_level.state == GridLevelStates.OPEN_ORDER_FILLED}"
                )

    def _place_dynamic_order(self, level: GridLevel):
        """
        Place an order on the specified level with dynamic side determination.
        """
        try:
            order_candidate = self._get_open_order_candidate(level)
            self.adjust_order_candidates(self.config.connector_name, [order_candidate])

            if order_candidate.amount > 0:
                order_id = self.place_order(
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    order_type=self.config.triple_barrier_config.open_order_type,
                    amount=order_candidate.amount,
                    price=order_candidate.price,
                    side=level.side,  # Use level's dynamic side
                    position_action=PositionAction.OPEN,
                )
                level.active_open_order = TrackedOrder(order_id=order_id)
                self.max_open_creation_timestamp = self._strategy.current_timestamp
                self.logger().info(
                    f"AutoFillDeb: Placed dynamic {level.side.name} order {order_id} at level {level.id} "
                    f"(price: {order_candidate.price}, amount: {order_candidate.amount})"
                )
            else:
                self.logger().warning(
                    f"AutoFillDeb: Cannot place order on level {level.id} - amount is 0"
                )
        except Exception as e:
            self.logger().error(
                f"AutoFillDeb: Failed to place dynamic order on level {level.id}: {e}"
            )

    def _update_grid_sides_based_on_price(self):
        """
        Update all grid level sides based on current market price.
        Levels above current price = SELL, levels below = BUY
        """
        current_price = self.mid_price

        for level in self.grid_levels:
            # Only update side for inactive levels to avoid disrupting active orders
            if level.state == GridLevelStates.NOT_ACTIVE:
                level.side = (
                    TradeType.SELL if level.price > current_price else TradeType.BUY
                )

    def process_order_canceled_event(
        self, _, market: ConnectorBase, event: OrderCancelledEvent
    ):
        """
        This method is responsible for processing the order canceled event and immediately replacing
        canceled orders with correct dynamic sides (like grid_strike behavior).
        Only replaces orders if the executor is still running (not shutting down).
        """
        self.update_grid_levels()
        levels_open_order_placed = [
            level for level in self.levels_by_state[GridLevelStates.OPEN_ORDER_PLACED]
        ]
        levels_close_order_placed = [
            level for level in self.levels_by_state[GridLevelStates.CLOSE_ORDER_PLACED]
        ]

        canceled_level = None

        for level in levels_open_order_placed:
            if event.order_id == level.active_open_order.order_id:
                self._canceled_orders.append(level.active_open_order.order_id)
                self.max_open_creation_timestamp = 0
                level.reset_open_order()  # Level becomes NOT_ACTIVE
                canceled_level = level
                break

        for level in levels_close_order_placed:
            if event.order_id == level.active_close_order.order_id:
                self._canceled_orders.append(level.active_close_order.order_id)
                self.max_close_creation_timestamp = 0
                level.reset_close_order()  # Level becomes OPEN_ORDER_FILLED

        if self._close_order and event.order_id == self._close_order.order_id:
            self._canceled_orders.append(self._close_order.order_id)
            self._close_order = None

        # Instant replacement logic (like grid_strike): immediately replace canceled open orders
        # ONLY if the executor is still running (not shutting down)
        if canceled_level is not None and self.status == RunnableStatus.RUNNING:
            self._instantly_replace_canceled_order(canceled_level)

    def _instantly_replace_canceled_order(self, level: GridLevel):
        """
        Instantly replace a canceled order with correct dynamic side (grid_strike behavior).
        This handles external cancellations from exchange UI.
        """
        current_price = self.mid_price

        # Determine correct side based on current price
        correct_side = TradeType.SELL if level.price > current_price else TradeType.BUY
        level.side = correct_side

        # Immediately place replacement order
        self.logger().info(
            f"Executor ID: {self.config.id} - Order canceled externally on level {level.id}, "
            f"instantly replacing with {correct_side.name} order"
        )
        self._place_dynamic_order(level)

    def process_order_failed_event(self, _, market, event: MarketOrderFailureEvent):
        """
        This method is responsible for processing the order failed event. Here we will add the InFlightOrder to the
        failed orders list.
        """
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

    def update_all_pnl_metrics(self):
        """
        Professional PnL calculation with proper position tracking and average cost basis.

        Logic:
        1. Process orders chronologically to maintain running position and average cost
        2. Each sell realizes PnL against current average cost basis
        3. Unrealized PnL = remaining position marked to current market price
        4. All volume tracking and metrics work as expected
        """
        self.logger().info("=== PROFESSIONAL PNL CALCULATION START ===")

        if len(self._filled_orders) == 0:
            self.logger().info("No filled orders - resetting all PnL metrics to zero")
            self._reset_all_metrics()
            return

        # DEBUG: Log available keys for troubleshooting
        if len(self._filled_orders) > 0:
            sample_order = self._filled_orders[0]
            self.logger().info(
                f"DEBUG: Available order keys: {list(sample_order.keys())}"
            )

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

            sorted_orders = sorted(self._filled_orders, key=get_fill_timestamp)

            self.logger().info(
                f"Processing {len(sorted_orders)} orders chronologically"
            )

            # Debug: Log order sequence with timestamps
            for i, order in enumerate(sorted_orders):
                side = order.get("trade_type", "UNKNOWN")
                price = order.get("price", 0)
                creation_ts = order.get("creation_timestamp", 0)
                update_ts = order.get("last_update_timestamp", 0)

                # Get actual fill timestamp
                fill_ts = get_fill_timestamp(order)

                self.logger().info(
                    f"DEBUG Order {i + 1}: {side} @ {price} (created: {creation_ts}, updated: {update_ts}, fill: {fill_ts})"
                )

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

                    self.logger().info(
                        f"Order {i + 1}: {trade_type} {float(executed_base):.6f} @ {float(avg_price):.4f} "
                        f"(fee: {float(order_fees):.4f})"
                    )

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

                                self.logger().info(
                                    f"  BUY Cover: {float(cover_amount):.6f} from short avg {float(avg_short_price):.4f} using {float(avg_price):.4f}, PnL {float(cover_pnl):.4f}"
                                )

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

                        # Finalize position size and average
                        position_size_base = new_position_size
                        if position_size_base > 0:
                            new_avg_cost = position_cost_total / position_size_base
                        elif position_size_base < 0:
                            new_avg_cost = position_cost_total / abs(position_size_base)
                        else:
                            new_avg_cost = Decimal("0")

                        self.logger().info(
                            f"  BUY: Position {float(prev_pos):.6f} -> {float(position_size_base):.6f}, Avg Cost: {float(new_avg_cost):.4f} (fees: {float(order_fees):.4f} tracked separately)"
                        )
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

                                self.logger().info(
                                    f"  SELL Reduce: {float(sell_from_long):.6f} from long avg {float(current_avg_cost):.4f} at {float(avg_price):.4f}, PnL {float(trade_realized_pnl):.4f}"
                                )

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

                        self.logger().info(
                            f"  SELL: Position {float(prev_pos):.6f} -> {float(new_position_size):.6f}, Avg {float(new_avg_cost):.4f} (fees: {float(order_fees):.4f})"
                        )

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

                    self.logger().info(
                        f"LONG position: {float(position_size_base):.6f} units @ {float(current_avg_cost):.4f} avg cost, "
                        f"Market value: {float(current_market_value):.4f}, Unrealized PnL: {float(unrealized_pnl):.4f}"
                    )

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

                    self.logger().info(
                        f"SHORT position: {float(abs(position_size_base)):.6f} units @ {float(current_avg_cost):.4f} avg cost, "
                        f"Market value: {float(current_market_value):.4f}, Unrealized PnL: {float(unrealized_pnl):.4f}"
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
                self.logger().info("No remaining position - all PnL is realized")
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
            net_pnl = self.realized_pnl_quote + self.position_pnl_quote

            self.logger().info("=== PROFESSIONAL PNL CALCULATION RESULTS ===")
            self.logger().info(
                "🎯 STANDARD APPROACH: Cost basis excludes fees, fees subtracted explicitly in PnL"
            )
            self.logger().info(
                f"📊 Orders Processed: {len(sorted_orders)} ({buy_count} buys, {sell_count} sells)"
            )
            self.logger().info(
                f"💰 Volume: Buy ${float(total_buy_volume):.4f}, Sell ${float(total_sell_volume):.4f}"
            )
            self.logger().info(
                f"💵 Realized PnL: ${float(self.realized_pnl_quote):.4f} (includes explicit fee deduction)"
            )
            self.logger().info(
                f"📈 Unrealized PnL: ${float(self.position_pnl_quote):.4f}"
            )
            self.logger().info(f"🎯 Net PnL: ${float(net_pnl):.4f}")
            self.logger().info(
                f"💸 Total Fees: ${float(total_fees):.4f} (tracked separately from cost basis)"
            )
            self.logger().info(
                f"📦 Position: {float(self.position_size_base):.6f} @ ${float(self.position_break_even_price):.4f} (pure avg cost)"
            )
            self.logger().info("✅ Cost basis now matches exchange UI standards!")
            self.logger().info("=== CALCULATION COMPLETE ===")
        except Exception as e:
            self.logger().error(
                f"Critical error in PnL calculation: {e}", exc_info=True
            )
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
        self.update_all_pnl_metrics()

    def update_realized_pnl_metrics(self):
        """Backward compatibility wrapper - now just calls the combined function."""
        self.update_all_pnl_metrics()

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
