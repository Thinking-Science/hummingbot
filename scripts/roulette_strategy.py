import datetime
import os
from decimal import Decimal
from typing import Dict, List

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
    def __init__(self, position_config: PositionConfig, strategy: ScriptStrategyBase, signal_value: int,
                 roulette_group_id: int = 1, ball_number: int = 1):
        super().__init__(position_config, strategy)
        self.signal_value = signal_value
        self.roulette_group_id = roulette_group_id
        self.ball_number = ball_number


class RouletteStrategy(ScriptStrategyBase):
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
    stored_executors: List[SignalExecutor] = []

    # Fee structure
    taker_fee = 0.0003
    maker_fee = 0.00012
    total_fee = taker_fee * 2

    # Configure the parameters for the position
    stop_loss_multiplier = 0.75
    take_profit_multiplier = 1.5
    time_limit = 60 * 55

    short_threshold = -0.5
    long_threshold = 0.5

    candles = CandlesFactory.get_candle(connector=exchange,
                                        trading_pair=trading_pair,
                                        interval="3m", max_records=500)

    # Configure the leverage and order amount the bot is going to use
    set_leverage_flag = None
    leverage = 20

    # Roulette amount configuration
    initial_order_amount_usd = Decimal("6")
    roulette_group_id = 1
    ball_number = 1

    today = datetime.datetime.today()
    csv_path = data_path() + f"/roulette_{exchange}_{trading_pair}_{today.day:02d}-{today.month:02d}-{today.year}-{today.hour}.csv"

    # Suscribe to trades
    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        # Is necessary to start the Candles Feed.
        super().__init__(connectors)
        self.candles.start()

    def on_stop(self):
        """
        Without this functionality, the network iterator will continue running forever after stopping the strategy
        That's why is necessary to introduce this new feature to make a custom stop with the strategy.
        """
        # we are going to close all the open positions when the bot stops
        self.close_open_positions()
        self.candles.stop()

    def get_active_executors(self):
        return self.active_executors

    def get_closed_executors(self):
        return self.stored_executors

    @property
    def all_candles_ready(self):
        """
        Checks if the candlesticks are full.
        """
        return self.candles.is_ready

    def on_tick(self):
        self.check_and_set_leverage()
        if len(self.get_active_executors()) < self.max_executors and self.all_candles_ready:
            signal, take_profit, stop_loss, indicators = self.get_signal_tp_and_sl()
            if signal < self.short_threshold or signal > self.long_threshold:
                bet = self.get_roulette_amount(take_profit)
                position_side = PositionSide.LONG if signal > 0 else PositionSide.SHORT
                self.notify_hb_app_with_timestamp(f"""
Creating new position for game {self.roulette_group_id} --> Ball: {self.ball_number}!
Signal: {signal} | {position_side}
Amount: {bet} | Take Profit: {take_profit} | Stop Loss: {stop_loss}
""")
                price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
                signal_executor = SignalExecutor(
                    position_config=PositionConfig(
                        timestamp=self.current_timestamp, trading_pair=self.trading_pair,
                        exchange=self.exchange, order_type=OrderType.MARKET,
                        side=position_side,
                        entry_price=price,
                        amount=bet / price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        time_limit=self.time_limit),
                    strategy=self,
                    signal_value=signal,
                    roulette_group_id=self.roulette_group_id,
                    ball_number=self.ball_number
                )
                self.active_executors.append(signal_executor)
        self.clean_and_store_executors()

    def get_signal_tp_and_sl(self):
        candles_df = self.candles.candles_df
        # Let's add some technical indicators
        candles_df.ta.bbands(length=100, append=True)
        candles_df.ta.macd(fast=21, slow=42, signal=9, append=True)
        candles_df["std"] = candles_df["close"].rolling(100).std()
        candles_df["std_close"] = candles_df["std"] / candles_df["close"]
        last_candle = candles_df.iloc[-1]
        bbp = last_candle["BBP_100_2.0"]
        macdh = last_candle["MACDh_21_42_9"]
        macd = last_candle["MACD_21_42_9"]
        std_pct = last_candle["std_close"]
        if bbp < 0.2 and macdh > 0 and macd < 0:
            signal_value = 1
        elif bbp > 0.8 and macdh < 0 and macd > 0:
            signal_value = -1
        else:
            signal_value = 0
        take_profit = std_pct * self.take_profit_multiplier
        stop_loss = std_pct * self.stop_loss_multiplier
        indicators = [bbp, macdh, macd]
        return signal_value, take_profit, stop_loss, indicators

    def get_executors_by_roulette_group_id(self):
        roulette_by_group_id = {}
        for executor in self.get_closed_executors():
            if executor.roulette_group_id not in roulette_by_group_id:
                roulette_by_group_id[executor.roulette_group_id] = [executor]
            roulette_by_group_id[executor.roulette_group_id].append(executor)
        return roulette_by_group_id

    def calculate_roulette_stats(self):
        results_by_roulette_id = {}
        for roullete_id, executors in self.get_executors_by_roulette_group_id().items():
            ball_numbers = len(executors)
            realized_pnl = sum([executor.pnl_usd for executor in executors])
            cum_fees = sum([executor.cum_fees for executor in executors])
            max_order_amount_usd = max([executor.amount * executor.entry_price for executor in executors])
            net_pnl = realized_pnl - cum_fees
            results_by_roulette_id[roullete_id] = {
                "ball_numbers": ball_numbers,
                "realized_pnl": realized_pnl,
                "cum_fees": cum_fees,
                "net_pnl": net_pnl,
                "max_order_amount_usd": max_order_amount_usd,
                "max_margin_usd": max_order_amount_usd / self.leverage,
            }
        return results_by_roulette_id

    def format_status(self) -> str:
        """
        Displays the three candlesticks involved in the script with RSI, BBANDS and EMA.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []

        pnl_usd = 0
        fees_cum_usd = 0

        if len(self.stored_executors) > 0:
            lines.extend(["\n########### Roulette Stats ###########"])
            for roulette_id, stats in self.calculate_roulette_stats().items():
                lines.extend([f"""
| Roulette id: {roulette_id} | Ball numbers: {stats['ball_numbers']} |
| Realized PNL: {stats['realized_pnl']:.4f} | Fees cum: {stats['cum_fees']:.4f} | Net result: {stats['net_pnl']:.4f} |
| Max order amount USD: {stats['max_order_amount_usd']:.4f} | Max margin USD: {stats['max_margin_usd']:.4f} |
"""])
        for executor in self.stored_executors:
            pnl_usd += executor.pnl_usd
            fees_cum_usd += executor.cum_fees
        if len(self.active_executors) > 0:
            lines.extend(["\n########### Active Executors ###########"])
            for executor in self.active_executors:
                lines.extend([f"|Signal id: {executor.timestamp} | Signal value: {executor.signal_value:.2f} | Ball number: {executor.ball_number} |"])
                lines.extend(executor.to_format_status())
                pnl_usd += executor.pnl_usd
                fees_cum_usd += executor.cum_fees

        lines.extend([f"\n| PNL USD: {pnl_usd:.4f} | Fees cum USD: {fees_cum_usd:.4f} | Net result: {(pnl_usd - fees_cum_usd):.4f} |"])

        if self.all_candles_ready:
            lines.extend(["\n############################################ Market Data ############################################\n"])
            signal, take_profit, stop_loss, indicators = self.get_signal_tp_and_sl()
            lines.extend([f"Signal: {signal} | Take Profit: {take_profit} | Stop Loss: {stop_loss}"])
            lines.extend([f"BB%: {indicators[0]} | MACDh: {indicators[1]} | MACD: {indicators[2]}"])
            lines.extend(["\n-----------------------------------------------------------------------------------------------------------\n"])
        else:
            lines.extend(["", "  No data collected."])
        return "\n".join(lines)

    def clean_and_store_executors(self):
        executors_to_store = [executor for executor in self.active_executors if executor.is_closed]
        if not os.path.exists(self.csv_path):
            df_header = pd.DataFrame([("timestamp",
                                       "roulette_group_id",
                                       "ball_number",
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
                                       "leverage",
                                       "open_order_id",
                                       "close_order_id",
                                       "realized_pnl_usd",
                                       "cum_fees_usd")])
            df_header.to_csv(self.csv_path, mode='a', header=False, index=False)
        for executor in executors_to_store:
            self.stored_executors.append(executor)
            close_order_id = executor.close_order.order_id if executor.close_order.order_id else ""
            df = pd.DataFrame([(executor.timestamp,
                                executor.executor_group_id,
                                executor.ball_number,
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
                                self.leverage,
                                executor.open_order.order_id,
                                close_order_id,
                                executor.pnl_usd,
                                executor.cum_fees
                                )])
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

    def get_roulette_amount(self, take_profit):
        net_loss_usd = 0
        ball_number = 1
        for executor in reversed(self.get_closed_executors()):
            if executor.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT:
                break
            else:
                ball_number += 1
                net_loss_usd += (executor.pnl_usd - executor.cum_fees)

        extra_amount = - net_loss_usd / Decimal(take_profit - self.taker_fee - self.maker_fee) if net_loss_usd < Decimal("0") else Decimal("0")

        if ball_number == 1 or extra_amount == 0:
            self.roulette_group_id += 1
            ball_number = 1
        self.ball_number = ball_number

        amount = self.initial_order_amount_usd + extra_amount

        if not self.is_margin_enough(betting_amount=amount):
            self.notify_hb_app_with_timestamp(f"Game over! Not enough margin to bet {amount}")
            amount = self.initial_order_amount_usd
            self.roulette_group_id += 1
            self.ball_number = 1
        return amount
