import asyncio
import uuid
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel

from hummingbot.core.data_type.common import OrderType, PositionSide
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.smart_components.position_executor.data_types import PositionConfig, PositionExecutorStatus
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor


class RouletteConfig(BaseModel):
    exchange: str
    trading_pair: str
    stop_loss_multiplier: Decimal
    take_profit_multiplier: Decimal
    time_limit: int
    max_stop_loss: Decimal
    trailing_stop_loss: bool = False
    trailing_stop_loss_pct: Decimal = Decimal("0")
    open_order_type: OrderType
    open_order_refresh_analyze_time: float = 30
    open_order_buffer_price: Decimal = Decimal("0.000001")
    max_balls: int = 7
    initial_order_amount: Decimal = Decimal("10.0")
    leverage: int = 20


class RouletteStatus(Enum):
    ACTIVE = 1
    PAUSED = 2
    CLOSED_BY_LOOT = 3
    CLOSED_BY_EARLY_LOOT = 4
    CLOSED_BY_GAME_OVER = 5
    CLOSED_BY_COMMAND = 6


class Roulette:
    def __init__(self, strategy, roulette_config: RouletteConfig):
        self.roulette_id = str(uuid.uuid4())
        self.ball_number = 1
        self._roulette_config: RouletteConfig = roulette_config
        self._strategy = strategy
        self.status = RouletteStatus.ACTIVE
        self.executors = []
        self.realized_net_pnl = Decimal("0")
        self.unrealized_net_pnl = Decimal("0")
        self.closed_by_early_loot = 0
        self.closed_by_stop_loss = 0
        self.closed_by_time_limit = 0
        self.canceled_by_time_limit = 0
        self.closed_by_loot = 0
        self.signal = 0
        self.std = Decimal("0")
        self.std_mean = Decimal("0")
        safe_ensure_future(self.control_loop())

    def is_closed(self):
        return self.status in [
            RouletteStatus.CLOSED_BY_LOOT,
            RouletteStatus.CLOSED_BY_EARLY_LOOT,
            RouletteStatus.CLOSED_BY_GAME_OVER,
            RouletteStatus.CLOSED_BY_COMMAND,
        ]

    async def control_loop(self):
        while not self.is_closed():
            self.control_roulette()
            await asyncio.sleep(1)

    def get_position_config(self, position_side, bet, price):
        return PositionConfig(
            timestamp=self._strategy.current_timestamp,
            trading_pair=self._roulette_config.trading_pair,
            exchange=self._roulette_config.exchange,
            order_type=self._roulette_config.open_order_type,
            side=position_side,
            entry_price=price,
            amount=bet / price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self._roulette_config.time_limit
        )

    @property
    def connector(self):
        return self._strategy.connectors[self._roulette_config.exchange]

    def control_roulette(self):
        self.update_status()
        if self.has_active_executors:
            self.check_refresh_open_order()
        elif not self.is_closed():
            if self.signal < self._strategy.short_threshold or self.signal > self._strategy.long_threshold:
                bet_usd = self.get_roulette_amount()
                price = self.connector.get_mid_price(self._roulette_config.trading_pair)
                position_side = PositionSide.LONG if self.signal > 0 else PositionSide.SHORT
                if self.is_margin_enough(bet_usd):
                    self.executors.append(PositionExecutor(self.get_position_config(position_side, bet_usd, price), self._strategy))
                else:
                    self.status = RouletteStatus.PAUSED
                    self._strategy.notify_hb_app_with_timestamp(f"{self._roulette_config.trading_pair} | Roulette {self.roulette_id} paused! Not enough margin... {bet_usd}")

    def get_roulette_amount(self):
        extra_amount = - self.realized_net_pnl / Decimal(self.take_profit - self._strategy.taker_fee - self._strategy.maker_fee)
        amount = self._roulette_config.initial_order_amount + extra_amount
        return amount

    @property
    def has_active_executors(self):
        return any([executor.status in [PositionExecutorStatus.ACTIVE_POSITION, PositionExecutorStatus.ORDER_PLACED] for executor in self.executors])

    def update_status(self):
        realized_net_pnl = 0
        unrealized_net_pnl = 0
        self.signal, self.std, self.std_mean = self._strategy.get_signal_std(self._roulette_config.trading_pair)
        for executor in reversed(self.executors):
            if executor.status == PositionExecutorStatus.ACTIVE_POSITION:
                unrealized_net_pnl += executor.pnl_usd - executor.cum_fees
            else:
                realized_net_pnl += executor.pnl_usd - executor.cum_fees
            if executor.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT:
                self.status = RouletteStatus.CLOSED_BY_LOOT
        if not self.has_active_executors:
            if -1 * realized_net_pnl > self.game_over_usd:
                self.status = RouletteStatus.CLOSED_BY_GAME_OVER
                self._strategy.notify_hb_app_with_timestamp(f"{self._roulette_config.trading_pair} | Roulette {self.roulette_id} closed by game over! Max loss reached... {realized_net_pnl}")
            elif self.status == RouletteStatus.CLOSED_BY_LOOT:
                self._strategy.notify_hb_app_with_timestamp(f"{self._roulette_config.trading_pair} | Roulette {self.roulette_id} closed by loot! {realized_net_pnl}")
            elif realized_net_pnl > 0:
                self.status = RouletteStatus.CLOSED_BY_EARLY_LOOT
                self._strategy.notify_hb_app_with_timestamp(f"{self._roulette_config.trading_pair} | Roulette {self.roulette_id} closed by early Loot! {realized_net_pnl}")

        self.realized_net_pnl = realized_net_pnl
        self.unrealized_net_pnl = unrealized_net_pnl
        self.canceled_by_time_limit = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CANCELED_BY_TIME_LIMIT])
        self.closed_by_time_limit = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CLOSED_BY_TIME_LIMIT])
        self.closed_by_loot = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT])
        self.closed_by_stop_loss = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CLOSED_BY_STOP_LOSS
                                        and executor.pnl_usd < Decimal("0")])
        self.closed_by_early_loot = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CLOSED_BY_STOP_LOSS
                                        and executor.pnl_usd > Decimal("0")])
        self.ball_number = len([executor for executor in self.executors if executor.status != PositionExecutorStatus.CANCELED_BY_TIME_LIMIT])

    @property
    def take_profit(self):
        return self.std * self._roulette_config.take_profit_multiplier

    @property
    def game_over_usd(self):
        cum_loss = 0
        for i in range(1, self._roulette_config.max_balls + 1, 1):
            take_profit_mean = self.std_mean * self._roulette_config.take_profit_multiplier
            extra_amount = Decimal(str(cum_loss)) / (take_profit_mean - self._strategy.taker_fee - self._strategy.maker_fee)
            cum_loss += (self._roulette_config.initial_order_amount + extra_amount) * self._roulette_config.max_stop_loss
        return cum_loss

    @property
    def stop_loss(self):
        return min(self._roulette_config.max_stop_loss, self._roulette_config.stop_loss_multiplier * self.std)

    def is_margin_enough(self, betting_amount):
        quote_balance = self.connector.get_balance(self._roulette_config.trading_pair.split("-")[-1])
        if betting_amount * Decimal("1.01") < quote_balance * self._roulette_config.leverage:
            return True
        else:
            self._strategy.logger().info("No enough margin to place orders.")
            return False

    def check_refresh_open_order(self):
        current_executor = self.executors[-1]
        if current_executor.status == PositionExecutorStatus.ORDER_PLACED:
            if self._strategy.current_timestamp - current_executor.position_config.timestamp > self._roulette_config.open_order_refresh_analyze_time:
                current_price = self.connector.get_mid_price(self._roulette_config.trading_pair)
                open_order_entry_price = current_executor._open_order._order.price
                buffer_used = 1 + self._roulette_config.open_order_buffer_price \
                    if current_executor.side == PositionSide.SHORT else 1 - self._roulette_config.open_order_buffer_price
                price_when_order_placed = Decimal(open_order_entry_price) / Decimal(buffer_used)
                if abs(Decimal(current_price) - Decimal(open_order_entry_price)) > abs(
                        Decimal(price_when_order_placed) - Decimal(open_order_entry_price)):
                    current_executor.cancel_executor_order_placed()
