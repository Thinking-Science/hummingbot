import asyncio
import uuid
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel

from hummingbot.core.data_type.common import OrderType
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.smart_components.position_executor.data_types import PositionConfig, PositionExecutorStatus
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor
from scripts.looter_king import LooterKing


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
    game_over_usd: Decimal = Decimal("7.0")
    initial_order_amount: Decimal = Decimal("10.0")
    leverage: Decimal = Decimal("20")


class RouletteStatus(Enum):
    ACTIVE = 1
    PAUSED = 2
    CLOSED_BY_LOOT = 3
    CLOSED_BY_EARLY_LOOT = 4
    CLOSED_BY_GAME_OVER = 5


class Roulette:
    def __init__(self, strategy: LooterKing, roulette_config: RouletteConfig):
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
        safe_ensure_future(self.control_loop())

    def is_closed(self):
        return self.status in [
            RouletteStatus.CLOSED_BY_LOOT,
            RouletteStatus.CLOSED_BY_EARLY_LOOT,
            RouletteStatus.CLOSED_BY_GAME_OVER,
        ]

    async def control_loop(self):
        while not self.is_closed():
            self.control_roulette()
            await asyncio.sleep(1)

    def control_roulette(self):
        if self.status == RouletteStatus.ACTIVE:
            self.check_roulette()
        elif self.status == RouletteStatus.PAUSED:
            self.resume_roulette()

    def get_position_config(self, position_side, bet, price):
        return PositionConfig(
            timestamp=self._strategy.current_timestamp,
            trading_pair=self._roulette_config.trading_pair,
            exchange=self._roulette_config.exchange,
            order_type=self._roulette_config.open_order_type,
            side=position_side,
            entry_price=price,
            amount=bet / price,
            stop_loss=self._roulette_config.stop_loss_multiplier,
            take_profit=self._roulette_config.take_profit_multiplier,
            time_limit=self._roulette_config.time_limit
        )

    @property
    def connector(self):
        return self._strategy.connectors[self._roulette_config.exchange]

    def check_roulette(self):
        self.update_status()
        if self.has_active_executors:
            return
        elif not self.is_closed():
            signal, std = self._strategy.get_signal_std(
                self._roulette_config.trading_pair)

            take_profit = std * self._roulette_config.take_profit_multiplier
            # stop_loss = min(self._roulette_config.max_stop_loss,
            #                 self._roulette_config.stop_loss_multiplier * std)

            if signal < self._strategy.short_threshold or signal > self._strategy.long_threshold:
                bet = self.get_roulette_amount(take_profit)
                price = self.connector.get_mid_price(self._roulette_config.trading_pair, True)
                bet_usd = bet * price
                if self.is_margin_enough(bet_usd):
                    self.executors.append(PositionExecutor(self._roulette_config, self._strategy))

    def get_roulette_amount(self, take_profit):
        extra_amount = - self.realized_net_pnl / Decimal(take_profit - self._strategy.taker_fee - self._strategy.maker_fee)
        amount = self._roulette_config.initial_order_amount + extra_amount
        return amount

    @property
    def has_active_executors(self):
        return any([executor.status in [PositionExecutorStatus.ACTIVE_POSITION, PositionExecutorStatus.ORDER_PLACED] for executor in self.executors])

    def update_status(self):
        realized_net_pnl = 0
        unrealized_net_pnl = 0
        for executor in reversed(self.executors):
            if executor.status == PositionExecutorStatus.ACTIVE_POSITION:
                unrealized_net_pnl += executor.pnl_usd - executor.cum_fees_usd
            else:
                realized_net_pnl += executor.pnl_usd - executor.cum_fees_usd
            if executor.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT:
                self.status = RouletteStatus.CLOSED_BY_LOOT
        if not self.has_active_executors:
            if -1 * realized_net_pnl > self._roulette_config.game_over_usd:
                self.status = RouletteStatus.CLOSED_BY_GAME_OVER
                self._strategy.notify_hb_app_with_timestamp(f"{self._roulette_config.trading_pair} | Roulette {self.roulette_id} closed by game over! Max loss reached... {realized_net_pnl}")
            elif self.status == RouletteStatus.CLOSED_BY_LOOT:
                self._strategy.notify_hb_app_with_timestamp(f"{self._roulette_config.trading_pair} | Roulette {self.roulette_id} closed by loot! {realized_net_pnl}")
            elif realized_net_pnl > 0:
                self.status = RouletteStatus.CLOSED_BY_EARLY_LOOT
                self._strategy.notify_hb_app_with_timestamp(f"{self._roulette_config.trading_pair} | Roulette {self.roulette_id} closed by early Loot! {realized_net_pnl}")

        self.realized_net_pnl = realized_net_pnl
        self.unrealized_net_pnl = unrealized_net_pnl
        self.ball_number = len(self.executors) + 1
        self.canceled_by_time_limit = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CANCELED_BY_TIME_LIMIT])
        self.closed_by_time_limit = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CLOSED_BY_TIME_LIMIT])
        self.closed_by_stop_loss = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CLOSED_BY_STOP_LOSS
                                        and executor.pnl_usd < Decimal("0")])
        self.closed_by_early_loot = len([executor for executor in self.executors if executor.status == PositionExecutorStatus.CLOSED_BY_STOP_LOSS
                                        and executor.pnl_usd > Decimal("0")])

    def is_margin_enough(self, betting_amount):
        quote_balance = self.connector.get_balance(self._roulette_config.trading_pair.split("-")[-1])
        if betting_amount * Decimal("1.01") < quote_balance * self._roulette_config.leverage:
            return True
        else:
            self.logger().info("No enough margin to place orders.")
            return False
