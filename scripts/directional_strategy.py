import datetime
import random
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import BaseModel

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.derivative.position import PositionSide
from hummingbot.core.data_type.common import OrderType, PositionMode
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.smart_components.position_executor.data_types import PositionConfig
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class Signal(BaseModel):
    id: int
    value: float
    position_config: PositionConfig


class BotProfile(BaseModel):
    balance_limit: Decimal
    max_order_amount: Decimal
    long_threshold: float
    short_threshold: float
    leverage: int


class SignalFactory:
    def __init__(self, max_records: int, connectors: Dict[str, ConnectorBase], interval: str = "1m"):
        self.connectors = connectors
        self.candles = {
            connector_name: {trading_pair: CandlesFactory.get_candle(connector="binance_spot",
                                                                     trading_pair=trading_pair,
                                                                     interval=interval, max_records=max_records)
                             for trading_pair in connector.trading_pairs} for
            connector_name, connector in self.connectors.items()}

    def start(self):
        for connector_name, trading_pairs_candles in self.candles.items():
            for candles in trading_pairs_candles.values():
                candles.start()

    def stop(self):
        for connector_name, trading_pairs_candles in self.candles.items():
            for candles in trading_pairs_candles.values():
                candles.stop()

    @property
    def all_data_sources_ready(self):
        return all(np.array([[candles.is_ready for trading_pair, candles in trading_pairs_candles.items()]
                             for connector_name, trading_pairs_candles in self.candles.items()]).flatten())

    def candles_df(self):
        return {connector_name: {trading_pair: candles.candles for trading_pair, candles in
                                 trading_pairs_candles.items()}
                for connector_name, trading_pairs_candles in self.candles.items()}

    def features_df(self):
        candles_df = self.candles_df().copy()
        for connector_name, trading_pairs_candles in candles_df.items():
            for trading_pair, candles in trading_pairs_candles.items():
                candles.ta.rsi(length=14, append=True)
        return candles_df

    def current_features(self):
        return {connector_name: {trading_pair: features.iloc[-1, :].to_dict() for trading_pair, features in
                                 trading_pairs_features.items()}
                for connector_name, trading_pairs_features in self.features_df().items()}

    def get_signals(self):
        if self.all_data_sources_ready:
            signals = self.current_features().copy()
            for connector_name, trading_pairs_features in signals.items():
                for trading_pair, features in trading_pairs_features.items():
                    value = (features["RSI_14"] - 50) / 50
                    signal = Signal(
                        id=str(random.randint(1, 1e10)),
                        value=value,
                        position_config=PositionConfig(
                            timestamp=datetime.datetime.now().timestamp(),
                            stop_loss=Decimal(0.0005),
                            take_profit=Decimal(0.006),
                            time_limit=60,
                            order_type=OrderType.MARKET,
                            amount=Decimal(1),
                            side=PositionSide.LONG if value < 0 else PositionSide.SHORT,
                            trading_pair=trading_pair,
                            exchange=connector_name,
                        ),
                    )
                    signals[connector_name][trading_pair] = signal
                    return signals
        else:
            return None


class DirectionalStrategyPerpetuals(ScriptStrategyBase):
    bot_profile = BotProfile(
        balance_limit=Decimal(1000),
        max_order_amount=Decimal(30),
        long_threshold=0.5,
        short_threshold=-0.5,
        leverage=10,
    )
    max_executors_by_connector_trading_pair = 1
    trading_pairs = ["ETH-USDT", "BTC-USDT"]
    exchange = "binance_perpetual_testnet"
    set_leverage_flag = None
    signal_executors: Dict[str, PositionExecutor] = {}
    stored_executors: List[str] = []
    markets = {exchange: set(trading_pairs)}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.signal_factory = None

    def on_stop(self):
        self.signal_factory.stop()

    def get_active_executors(self):
        return {signal: executor for signal, executor in self.signal_executors.items() if not executor.is_closed}

    def get_active_executors_by_connector_trading_pair(self, connector_name, trading_pair):
        return {signal: executor for signal, executor in self.signal_executors.items() if not executor.is_closed
                and executor.exchange == connector_name
                and executor.trading_pair == trading_pair
                }

    def get_closed_executors(self):
        return {signal: executor for signal, executor in self.signal_executors.items() if executor.is_closed}

    def get_active_positions_df(self):
        active_positions = []
        for connector_name, connector in self.connectors.items():
            for trading_pair, position in connector.account_positions.items():
                active_positions.append({
                    "exchange": connector_name,
                    "trading_pair": trading_pair,
                    "side": position.position_side,
                    "entry_price": position.entry_price,
                    "amount": position.amount,
                    "leverage": position.leverage,
                    "unrealized_pnl": position.unrealized_pnl
                })
        return pd.DataFrame(active_positions)

    def on_tick(self):
        if not self.set_leverage_flag:
            for connector in self.connectors.values():
                for trading_pair in connector.trading_pairs:
                    connector.set_position_mode(PositionMode.HEDGE)
                    connector.set_leverage(trading_pair=trading_pair, leverage=self.bot_profile.leverage)
            self.set_leverage_flag = True
        if not self.signal_factory:
            self.signal_factory = SignalFactory(max_records=500, connectors=self.connectors)
            self.signal_factory.start()
        # TODO: Order the dictionary by highest abs signal values
        if self.signal_factory.all_data_sources_ready:
            for connector_name, trading_pair_signals in self.signal_factory.get_signals().items():
                for trading_pair, signal in trading_pair_signals.items():
                    if len(self.get_active_executors_by_connector_trading_pair(connector_name, trading_pair).keys()) < self.max_executors_by_connector_trading_pair:
                        if signal.value > self.bot_profile.long_threshold or signal.value < self.bot_profile.short_threshold:
                            position_config = signal.position_config
                            price = self.connectors[position_config.exchange].get_mid_price(position_config.trading_pair)
                            position_config.amount = (self.bot_profile.max_order_amount / price) * position_config.amount
                            self.signal_executors[signal.id] = PositionExecutor(
                                position_config=position_config,
                                strategy=self
                            )

        self.store_executors()

    def store_executors(self):
        # TODO: add options to store in database or csv.
        pass

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        # Show active positions
        positions_df = self.get_active_positions_df()
        if not positions_df.empty:
            lines.extend(
                ["", "  Positions:"] + ["    " + line for line in positions_df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active positions."])

        if len(self.get_closed_executors().keys()) > 0:
            lines.extend(["\n########################################## Closed Executors ##########################################"])

        for signal_id, executor in self.get_closed_executors().items():
            lines.extend([f"|Signal id: {signal_id}"])
            lines.extend(executor.to_format_status())
            lines.extend(["-----------------------------------------------------------------------------------------------------------"])

        if len(self.get_active_executors().keys()) > 0:
            lines.extend(["\n########################################## Active Executors ##########################################"])

        for signal_id, executor in self.get_active_executors().items():
            lines.extend([f"|Signal id: {signal_id}"])
            lines.extend(executor.to_format_status())
        if self.signal_factory and self.signal_factory.all_data_sources_ready:
            lines.extend(["\n############################################ Market Data ############################################"])
            candles_df = self.signal_factory.features_df()
            for connector_name, trading_pair_signal in self.signal_factory.get_signals().items():
                for trading_pair, signal in trading_pair_signal.items():
                    df = candles_df[self.exchange][trading_pair]
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    lines.extend([f"""

| Trading Pair: {trading_pair} | Exchange: {connector_name}
| Signal: {signal.value:.2f}

"""])
                    lines.extend(["    " + line for line in df.tail().to_string(index=False).split("\n")])
                    lines.extend(["\n-----------------------------------------------------------------------------------------------------------"])

        else:
            lines.extend(["", "  No data collected."])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
