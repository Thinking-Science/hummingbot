import datetime
import os
from collections import deque
from decimal import Decimal
from typing import Deque, Dict, List

import pandas as pd
import pandas_ta as ta  # noqa: F401

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.smart_components.position_executor.data_types import PositionConfig, PositionExecutorStatus
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class SignalExecutor(PositionExecutor):
    def __init__(self, position_config: PositionConfig, strategy: ScriptStrategyBase, signal_value: int):
        super().__init__(position_config, strategy)
        self.signal_value = signal_value


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
    active_executors: List[SignalExecutor] = []
    stored_executors: Deque[SignalExecutor] = deque(maxlen=10)  # Store only the last 10 executors for reporting

    # Configure the parameters for the position
    taker_fee = 0.0003
    maker_fee = 0.00012
    total_fee = taker_fee * 2
    take_profit = 0.007
    stop_loss = 0.002
    time_limit = 60 * 30
    real_take_profit = take_profit - taker_fee - maker_fee

    short_threshold = -0.5
    long_threshold = 0.5

    position_side = PositionSide.SHORT
    signal_side = PositionSide.SHORT

    # Create the candles that we want to use and the thresholds for the indicators
    candles_1m = CandlesFactory.get_candle(connector=exchange,
                                           trading_pair=trading_pair,
                                           interval="1m", max_records=500)
    candles_3m = CandlesFactory.get_candle(connector=exchange,
                                           trading_pair=trading_pair,
                                           interval="3m", max_records=500)
    candles = {
        f"{trading_pair}_1m": candles_1m,
        f"{trading_pair}_3m": candles_3m,
    }

    # Configure the leverage and order amount the bot is going to use
    set_leverage_flag = None
    leverage = 15
    initial_order_amount_usd = Decimal("10")
    order_amount_usd = Decimal("10")

    max_net_loss_usd = 10

    today = datetime.datetime.today()
    csv_path = data_path() + f"/{exchange}_{trading_pair}_{today.day:02d}-{today.month:02d}-{today.year}.csv"
    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        # Is necessary to start the Candles Feed.
        super().__init__(connectors)
        for candle in self.candles.values():
            candle.start()

    def get_active_executors(self):
        return self.active_executors

    def get_closed_executors(self):
        return self.stored_executors

    @property
    def all_candles_ready(self):
        """
        Checks if the candlesticks are full.
        """
        return all([candle.is_ready for candle in self.candles.values()])

    def on_tick(self):
        self.check_and_set_leverage()
        if len(self.get_active_executors()) < self.max_executors and self.all_candles_ready:
            signal = self.get_signal()
            if signal < self.short_threshold or signal > self.long_threshold:
                bet = self.get_roulette_amount()
                if not self.is_margin_enough(betting_amount=bet):
                    self.logger().info("Game over, max_net_loss_usd reached!")
                    bet = self.initial_order_amount_usd

                price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
                signal_executor = SignalExecutor(
                    position_config=PositionConfig(
                        timestamp=self.current_timestamp, trading_pair=self.trading_pair,
                        exchange=self.exchange, order_type=OrderType.MARKET,
                        side=PositionSide.LONG if signal > 0 else PositionSide.SHORT,
                        entry_price=price,
                        amount=bet / price,
                        stop_loss=self.stop_loss,
                        take_profit=self.take_profit,
                        time_limit=self.time_limit),
                    strategy=self,
                    signal_value=signal
                )
                self.active_executors.append(signal_executor)
        self.clean_and_store_executors()

    def get_signal(self):
        values = {}
        for candle_name, candle in self.candles.items():
            candle_df = candle.candles_df
            candle_df.ta.sma(length=7, append=True)
            candle_df.ta.sma(length=25, append=True)
            candle_df.ta.sma(length=99, append=True)
            last_row = candle_df.iloc[-1]
            sma_7 = last_row["SMA_7"].item()
            sma_25 = last_row["SMA_25"].item()
            sma_99 = last_row["SMA_99"].item()
            if sma_7 > sma_25 > sma_99:
                side = PositionSide.LONG
            elif sma_7 < sma_25 < sma_99:
                side = PositionSide.SHORT
            else:
                side = PositionSide.BOTH
            values[candle_name] = side

        if values[f"{self.trading_pair}_1m"] == PositionSide.LONG and values[f"{self.trading_pair}_3m"] in [PositionSide.LONG, PositionSide.BOTH]:
            composed_signal_value = 1
        elif values[f"{self.trading_pair}_1m"] == PositionSide.SHORT and values[f"{self.trading_pair}_3m"] in [PositionSide.SHORT, PositionSide.BOTH]:
            composed_signal_value = -1
        else:
            composed_signal_value = 0
        return composed_signal_value

    def on_stop(self):
        """
        Without this functionality, the network iterator will continue running forever after stopping the strategy
        That's why is necessary to introduce this new feature to make a custom stop with the strategy.
        """
        # we are going to close all the open positions when the bot stops
        self.close_open_positions()
        for candle in self.candles.values():
            candle.stop()

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
                lines.extend([f"|Signal id: {executor.timestamp} | Signal value: {executor.signal_value:.2f}"])
                lines.extend(executor.to_format_status())
        if self.all_candles_ready:
            lines.extend([
                "\n############################################ Market Data ############################################\n"])
            values = {}
            columns_to_show = ["timestamp", "open", "low", "high", "close", "volume", "SMA_7", "SMA_25", "SMA_99"]
            for candle_name, candles in self.candles.items():
                candles_df = candles.candles_df
                # Let's add some technical indicators
                # candles_df.ta.bbands(length=21, append=True)
                # candles_df.ta.rsi(length=21, append=True)
                # candles_df.ta.sma(length=10, close="RSI_21", prefix="RSI_21", append=True)
                candles_df.ta.sma(length=7, append=True)
                candles_df.ta.sma(length=25, append=True)
                candles_df.ta.sma(length=99, append=True)

                candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], unit="ms")
                lines.extend([f"Candles: {candles.name} | Interval: {candles.interval}\n"])
                lines.extend(
                    ["    " + line for line in
                     candles_df[columns_to_show].tail().to_string(index=False).split("\n")])

                # Signal Genaration
                last_row = candles_df.iloc[-1]
                sma_7 = last_row["SMA_7"].item()
                sma_25 = last_row["SMA_25"].item()
                sma_99 = last_row["SMA_99"].item()
                if sma_7 > sma_25 > sma_99:
                    side = PositionSide.LONG
                elif sma_7 < sma_25 < sma_99:
                    side = PositionSide.SHORT
                else:
                    side = PositionSide.BOTH
                lines.extend([f"Side: {side}"])
                values[candle_name] = side
            lines.extend(["\n-----------------------------------------------------------------------------------------------------------\n"])
        else:
            lines.extend(["", "  No data collected."])
        return "\n".join(lines)

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

    def is_margin_enough(self, betting_amount):
        quote_balance = self.connectors[self.exchange].get_balance(self.trading_pair.split("-")[-1])
        if betting_amount * Decimal("1.01") < quote_balance * Decimal(str(self.leverage)):
            return True
        else:
            self.logger().info("No enough margin to place orders.")
            return False

    def check_and_set_leverage(self):
        if not self.set_leverage_flag:
            for connector in self.connectors.values():
                for trading_pair in connector.trading_pairs:
                    connector.set_position_mode(PositionMode.HEDGE)
                    connector.set_leverage(trading_pair=trading_pair, leverage=self.leverage)
            self.set_leverage_flag = True

    def get_roulette_amount(self):
        net_loss_usd = 0
        for executor in reversed(self.get_closed_executors()):
            if executor.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT:
                break
            else:
                if executor.status == PositionExecutorStatus.CLOSED_BY_STOP_LOSS:
                    exit_fee = executor.stop_loss_order.executed_amount_base * Decimal(str(self.taker_fee))
                elif executor.status == PositionExecutorStatus.CLOSED_BY_TIME_LIMIT:
                    exit_fee = executor.time_limit_order.executed_amount_base * Decimal(str(self.taker_fee))
                enter_fee = (executor.amount * Decimal(str(self.taker_fee)))
                realized_pnl = Decimal(str(executor.pnl)) * executor.amount
                net_loss_usd += (realized_pnl - enter_fee - exit_fee) * executor.entry_price

        amount = self.initial_order_amount_usd - net_loss_usd / Decimal(self.real_take_profit)
        self.logger().info(f'net_loss_usd: {net_loss_usd}')
        self.logger().info(f'Amount based in net_loss_usd: {amount}')
        return amount
