import datetime
import os
from collections import deque
from decimal import Decimal
from typing import Deque, List

import pandas as pd
import pandas_ta as ta  # noqa: F401

from hummingbot import data_path
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide
from hummingbot.smart_components.position_executor.data_types import PositionConfig
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class SimpleDirectionalStrategyExample(ScriptStrategyBase):
    """
    A simple trading strategy that uses RSI in one timeframe to determine whether to go long or short.
    IMPORTANT: Binance perpetual has to be in Single Asset Mode, soon we are going to support Multi Asset Mode.
    """
    # Define the trading pair and exchange that we want to use and the csv where we are going to store the entries
    trading_pair = "DODO-BUSD"
    exchange = "binance_perpetual"

    # Maximum position executors at a time
    max_executors = 1
    active_executors: List[PositionExecutor] = []
    stored_executors: Deque[PositionExecutor] = deque(maxlen=10)  # Store only the last 10 executors for reporting

    # Configure the parameters for the position
    fee = 0.002
    stop_loss = fee
    take_profit = 3 * fee
    time_limit = 60 * 30

    # Create the candles that we want to use and the thresholds for the indicators
    # eth_3m_candles = CandlesFactory.get_candle(connector=exchange,
    #                                            trading_pair=trading_pair,
    #                                            interval="1m", max_records=50)

    # Configure the leverage and order amount the bot is going to use
    starting_leverage = 1
    leverage = 1
    initial_order_amount_usd = Decimal("10")
    order_amount_usd = Decimal("10")

    max_cum_failures = 6

    today = datetime.datetime.today()
    csv_path = data_path() + f"/{exchange}_{trading_pair}_{today.day:02d}-{today.month:02d}-{today.year}.csv"
    markets = {exchange: {trading_pair}}

    # def __init__(self, connectors: Dict[str, ConnectorBase]):
    #     # Is necessary to start the Candles Feed.
    #     super().__init__(connectors)
    #     self.eth_3m_candles.start()

    def get_active_executors(self):
        return self.active_executors

    def get_closed_executors(self):
        return self.stored_executors

    def on_tick(self):
        if len(self.get_active_executors()) < self.max_executors:
            self.check_cum_failures_and_set_leverage()
            position_side = PositionSide.SHORT
            price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
            if self.is_margin_enough():
                signal_executor = PositionExecutor(
                    position_config=PositionConfig(
                        timestamp=self.current_timestamp, trading_pair=self.trading_pair,
                        exchange=self.exchange, order_type=OrderType.MARKET,
                        side=position_side,
                        entry_price=price,
                        amount=self.order_amount_usd / price,
                        stop_loss=self.stop_loss,
                        take_profit=self.take_profit,
                        time_limit=self.time_limit),
                    strategy=self,
                )
                self.active_executors.append(signal_executor)
        self.clean_and_store_executors()

    # def get_signal(self):
    #     candle_df = self.eth_3m_candles.candles_df
    #     # Let's add some technical indicators
    #     candle_df.ta.rsi(length=21, append=True)
    #     rsi_value = candle_df.iat[-1, -1]
    #     return rsi_value

    def on_stop(self):
        """
        Without this functionality, the network iterator will continue running forever after stopping the strategy
        That's why is necessary to introduce this new feature to make a custom stop with the strategy.
        """
        # we are going to close all the open positions when the bot stops
        self.close_open_positions()
        # self.eth_3m_candles.stop()

    def format_status(self) -> str:
        """
        Displays the three candlesticks involved in the script with RSI, BBANDS and EMA.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []

        if len(self.stored_executors) > 0:
            lines.extend([
                "\n########################################## Closed Executors ##########################################"])

        for executor in self.stored_executors:
            lines.extend([f"|Signal id: {executor.timestamp}"])
            lines.extend(executor.to_format_status())
            lines.extend([
                "-----------------------------------------------------------------------------------------------------------"])

        if len(self.active_executors) > 0:
            lines.extend([
                "\n########################################## Active Executors ##########################################"])

        for executor in self.active_executors:
            lines.extend([f"|Signal id: {executor.timestamp}"])
            lines.extend(executor.to_format_status())
        # if self.eth_3m_candles.is_ready:
        #     lines.extend([
        #         "\n############################################ Market Data ############################################\n"])
        #     lines.extend([f"Value: {self.get_signal()}"])
        #     columns_to_show = ["timestamp", "open", "low", "high", "close", "volume", "RSI_21"]
        #     candles_df = self.eth_3m_candles.candles_df
        #     # Let's add some technical indicators
        #     candles_df.ta.rsi(length=21, append=True)
        #     candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], unit="ms")
        #     lines.extend([f"Candles: {self.eth_3m_candles.name} | Interval: {self.eth_3m_candles.interval}\n"])
        #     lines.extend(["    " + line for line in candles_df[columns_to_show].tail().to_string(index=False).split("\n")])
        #     lines.extend(["\n-----------------------------------------------------------------------------------------------------------\n"])
        # else:
        #     lines.extend(["", "  No data collected."])

        return "\n".join(lines)

    def check_cum_failures_and_set_leverage(self):
        cum_failures = 0
        for executor in reversed(self.get_closed_executors()):
            if executor.pnl < 0:
                cum_failures += 1
            else:
                break
        for connector in self.connectors.values():
            for trading_pair in connector.trading_pairs:
                connector.set_position_mode(PositionMode.HEDGE)
                # TODO: change logic to run this strategy in multiple exchanges and trading pairs
                if cum_failures > self.max_cum_failures:
                    self.logger().info("Max leverage reached, starting over. To your home")
                    cum_failures = 0
                self.leverage = self.starting_leverage * 2**cum_failures
                self.order_amount_usd = self.initial_order_amount_usd * self.leverage
                connector.set_leverage(trading_pair=trading_pair, leverage=self.leverage)

    def clean_and_store_executors(self):
        executors_to_store = [executor for executor in self.active_executors if executor.is_closed]
        if not os.path.exists(self.csv_path):
            df_header = pd.DataFrame([("timestamp",
                                       "exchange",
                                       "trading_pair",
                                       "side",
                                       "amount",
                                       "pnl",
                                       "close_timestamp",
                                       "entry_price",
                                       "close_price",
                                       "last_status",
                                       "sl",
                                       "tp",
                                       "tl",
                                       "order_type",
                                       "leverage")])
            df_header.to_csv(self.csv_path, mode='a', header=False, index=False)
        for executor in executors_to_store:
            self.stored_executors.append(executor)
            df = pd.DataFrame([(executor.timestamp,
                                executor.exchange,
                                executor.trading_pair,
                                executor.side,
                                executor.amount,
                                executor.pnl,
                                executor.close_timestamp,
                                executor.entry_price,
                                executor.close_price,
                                executor.status,
                                executor.position_config.stop_loss,
                                executor.position_config.take_profit,
                                executor.position_config.time_limit,
                                executor.open_order_type,
                                self.leverage)])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        self.active_executors = [executor for executor in self.active_executors if not executor.is_closed]

    def close_open_positions(self):
        # we are going to close all the open positions when the bot stops
        for connector_name, connector in self.connectors.items():
            for trading_pair, position in connector.account_positions.items():
                if position.position_side == PositionSide.LONG:
                    self.sell(connector_name=connector_name,
                              trading_pair=position.trading_pair,
                              amount=abs(position.amount),
                              order_type=OrderType.MARKET,
                              price=connector.get_mid_price(position.trading_pair),
                              position_action=PositionAction.CLOSE)
                elif position.position_side == PositionSide.SHORT:
                    self.buy(connector_name=connector_name,
                             trading_pair=position.trading_pair,
                             amount=abs(position.amount),
                             order_type=OrderType.MARKET,
                             price=connector.get_mid_price(position.trading_pair),
                             position_action=PositionAction.CLOSE)

    def is_margin_enough(self):
        quote_balance = self.connectors[self.exchange].get_available_balance(self.trading_pair.split("-")[-1])
        if self.order_amount_usd < quote_balance * self.leverage:
            return True
        else:
            self.logger().info("No enough margin to place orders.")
            return False
