import datetime
import pickle
import random
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import BaseModel

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.derivative.position import PositionSide
from hummingbot.connector.markets_recorder import MarketsRecorder
from hummingbot.core.data_type.common import OrderType, PositionMode
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.features.features import Features
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
    def __init__(self, max_records: int, connectors: Dict[str, ConnectorBase], interval: str = "1m", features=Features()):
        self.connectors = connectors
        self.model = pickle.load(open('models/pipeline.pkl', 'rb'))
        self.candles = {
            connector_name: {trading_pair: CandlesFactory.get_candle(connector="binance_spot",
                                                                     trading_pair=trading_pair,
                                                                     interval=interval, max_records=max_records)
                             for trading_pair in connector.trading_pairs} for
            connector_name, connector in self.connectors.items()}
        self.features_dict = {
            'crossover': {
                'rsi': [
                    {'upper_thold': 70, 'upper_side': 'SHORT', 'down_thold': 30, 'down_side': 'LONG',
                     'config': {'length': 14}},
                    {'upper_thold': 65, 'upper_side': 'SHORT', 'down_thold': 35, 'down_side': 'LONG',
                     'config': {'length': 21}},
                    {'upper_thold': 60, 'upper_side': 'SHORT', 'down_thold': 40, 'down_side': 'LONG',
                     'config': {'length': 28}},
                ],
                'bbands': [
                    {'upper_side': 'SHORT', 'down_side': 'LONG', 'config': {'length': 20}},
                    {'upper_side': 'SHORT', 'down_side': 'LONG', 'config': {'length': 30, 'mamode': 't3'}},
                ]
            },
            'intersection': {
                'macd': [
                    {'series1_breakup_side': 'LONG', 'series2_breakup_side': 'SHORT',
                     'config': {"fast": 12, "slow": 26, "signal": 9}},
                    {'series1_breakup_side': 'LONG', 'series2_breakup_side': 'SHORT',
                     'config': {"fast": 15, "slow": 30, "signal": 12}},
                    {'series1_breakup_side': 'LONG', 'series2_breakup_side': 'SHORT',
                     'config': {"fast": 18, "slow": 34, "signal": 15}},
                ]
            }
        }
        self.features = features

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
            for trading_pair, df in trading_pairs_candles.items():
                df = self.features.add_features(df, self.features_dict, dropna=True)
        return candles_df

    def current_features(self):
        return {connector_name: {trading_pair: features.iloc[-2:, :] for trading_pair, features in
                                 trading_pairs_features.items()}
                for connector_name, trading_pairs_features in self.features_df().items()}

    def get_signals(self):
        signals = self.current_features().copy()
        for connector_name, trading_pairs_features in signals.items():
            for trading_pair, features in trading_pairs_features.items():
                # features = pd.DataFrame.from_dict(features)
                predict = self.model.predict(features)[-1]
                predict_proba = self.model.predict_proba(features).max()
                if predict == 0:
                    value = 0
                elif predict == 1:
                    value = predict_proba
                else:
                    value = -predict_proba
                width = 0.5
                target = features.close.pct_change().ewm(span=100).std().iat[-1] * width
                signal = Signal(
                    id=str(random.randint(1, 1e10)),
                    value=Decimal(str(value)),
                    position_config=PositionConfig(
                        timestamp=datetime.datetime.now().timestamp(),
                        stop_loss=Decimal(target),
                        take_profit=Decimal(target),
                        time_limit=60,
                        order_type=OrderType.MARKET,
                        amount=Decimal(1),
                        side=PositionSide.LONG if value > 0 else PositionSide.SHORT,
                        trading_pair=trading_pair,
                        exchange=connector_name,
                    ),
                )
                signals[connector_name][trading_pair] = signal
        return signals


class DirectionalStrategyPerpetuals(ScriptStrategyBase):
    bot_profile = BotProfile(
        balance_limit=Decimal(1000),
        max_order_amount=Decimal(20),
        long_threshold=0.5,
        short_threshold=-0.5,
        leverage=10,
    )
    max_executors_by_connector_trading_pair = 1
    trading_pairs = ["ETH-BUSD"]
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
        executors_to_store = {signal_id: executor for signal_id, executor in self.get_closed_executors().items()
                              if signal_id not in self.stored_executors}
        for signal_id, executor in executors_to_store.items():
            signal = {
                "id": signal_id,
                "timestamp": int(executor.timestamp),
                "close_timestamp": int(executor.close_timestamp),
                "sl": executor.position_config.stop_loss,
                "tp": executor.position_config.take_profit,
                "tl": executor.position_config.time_limit,
                "exchange": executor.exchange,
                "trading_pair": executor.trading_pair,
                "side": executor.side.name,
                "last_status": executor.status.name,
                "order_type": executor.position_config.order_type.name,
                "amount": executor.amount,
                "entry_price": executor.entry_price,
                "close_price": executor.close_price,
                "pnl": executor.pnl,
                "leverage": self.bot_profile.leverage,
            }
            MarketsRecorder.get_instance().add_closed_signal(signal)
            self.stored_executors.append(signal_id)

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
