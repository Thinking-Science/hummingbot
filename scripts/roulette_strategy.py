import datetime
import os
from decimal import Decimal
from typing import Dict

import pandas as pd
import pandas_ta as ta  # noqa: F401

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide, PriceType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.smart_components.position_executor.data_types import PositionConfig, PositionExecutorStatus
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class SignalExecutor(PositionExecutor):
    def __init__(self, position_config: PositionConfig, strategy: ScriptStrategyBase, signal_value: int,
                 roulette_group_id: int = 0, ball_number: int = 1):
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
    trading_pairs = ["DODO-BUSD", "APE-BUSD"]
    exchange = "binance_perpetual"

    today = datetime.datetime.today()

    # Maximum position executors at a time
    max_executors = 1
    roulette_by_trading_pair = {}
    for trading_pair in trading_pairs:
        roulette_by_trading_pair[trading_pair] = {
            "stop_loss_multiplier": 0.5,
            "max_stop_loss": 0.005,
            "take_profit_multiplier": 1,
            "time_limit": 60 * 55,
            "order_placed_time_limt": 1,
            "limit_order_price_buffer": 0.001,
            "leverage": 20,
            "initial_order_amount_usd": Decimal("6"),
            "active_executors": [],
            "stored_executors": [],
            "candles": CandlesFactory.get_candle(connector=exchange,
                                                 trading_pair=trading_pair,
                                                 interval="1m", max_records=500),
            "roulette_group_id": 0,
            "ball_number": 1,
            "csv_path": data_path() + f"/roulette_{exchange}_{trading_pair}_{today.day:02d}-{today.month:02d}-{today.year}-{today.hour}.csv"}

    # Fee structure
    taker_fee = 0.0003
    maker_fee = 0.00012
    total_fee = taker_fee * 2

    short_threshold = -0.5
    long_threshold = 0.5

    # Configure the leverage and order amount the bot is going to use
    set_leverage_flag = None

    # Suscribe to trades
    markets = {exchange: set(trading_pairs)}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        # Is necessary to start the Candles Feed.
        super().__init__(connectors)
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            roulette_info["candles"].start()

    def on_stop(self):
        """
        Without this functionality, the network iterator will continue running forever after stopping the strategy
        That's why is necessary to introduce this new feature to make a custom stop with the strategy.
        """
        # we are going to close all the open positions when the bot stops
        self.close_open_positions()
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            roulette_info["candles"].stop()

    def on_tick(self):
        self.clean_and_store_executors() #Stores closed position and refresh orders waiting to enter
        self.check_and_set_leverage()
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            if len(roulette_info["active_executors"]) < self.max_executors and roulette_info["candles"].is_ready:
                signal, take_profit, stop_loss, indicators = self.get_signal_tp_and_sl(roulette_info)
                if signal < self.short_threshold or signal > self.long_threshold:
                    bet = self.get_roulette_amount(trading_pair, roulette_info, take_profit) #sets ball number - roullete_id - amount+extraamount
                    position_side = PositionSide.LONG if signal > 0 else PositionSide.SHORT
                    price = self.get_entry_price(trading_pair,roulette_info,position_side) #gets best bid/ask price * buffer
                    self.logger().info(f"""
Trading Pair: {trading_pair}
Creating new position for game {roulette_info['roulette_group_id']} --> Ball: {roulette_info['ball_number']}!
Signal: {signal} | {position_side}
Amount: {bet} | Take Profit: {take_profit} | Stop Loss: {stop_loss}
                    """)
                    signal_executor = SignalExecutor(
                        position_config=PositionConfig(
                            timestamp=self.current_timestamp, trading_pair=trading_pair,
                            exchange=self.exchange, order_type=OrderType.LIMIT,
                            side=position_side,
                            entry_price=price,
                            amount=bet / price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            time_limit=roulette_info["time_limit"]),
                        strategy=self,
                        signal_value=signal,
                        roulette_group_id=roulette_info["roulette_group_id"],
                        ball_number=roulette_info["ball_number"]
                    )
                    roulette_info["active_executors"].append(signal_executor)

    def get_signal_tp_and_sl(self, roulette_info):
        candles_df = roulette_info["candles"].candles_df
        # Let's add some technical indicators
        candles_df.ta.bbands(length=100, append=True)
        candles_df.ta.macd(fast=21, slow=42, signal=9, append=True)
        candles_df["std"] = candles_df["close"].rolling(100).std()
        candles_df["mean"] = candles_df["close"].rolling(100).mean()
        candles_df["std_close"] = candles_df["std"] / candles_df["close"]
        last_candle = candles_df.iloc[-1]
        bbp = last_candle["BBP_100_2.0"]
        macdh = last_candle["MACDh_21_42_9"]
        macd = last_candle["MACD_21_42_9"]
        std_pct = last_candle["std_close"]
        if bbp < 0.7 and macdh > 0 and macd < 0:
            signal_value = 1
        elif bbp > 0.3 and macdh < 0 and macd > 0:
            signal_value = -1
        else:
            signal_value = 0

        take_profit = std_pct * roulette_info["take_profit_multiplier"]
        stop_loss = min(roulette_info["stop_loss_multiplier"], std_pct * 0.5)
        indicators = [bbp, macdh, macd]
        return signal_value, take_profit, stop_loss, indicators

    def get_executors_by_roulette_group_id(self, roulette_info):
        roulette_by_group_id = {}
        for executor in roulette_info["stored_executors"]:
            if executor.roulette_group_id not in roulette_by_group_id:
                roulette_by_group_id[executor.roulette_group_id] = [executor]
            else:
                roulette_by_group_id[executor.roulette_group_id].append(executor)
        return roulette_by_group_id

    def calculate_roulette_stats(self, roulette_info):
        results_by_roulette_id = {}
        for roullete_id, executors in self.get_executors_by_roulette_group_id(roulette_info).items():
            ball_numbers = len([x for x in executors if x.status != PositionExecutorStatus.CANCELED_BY_TIME_LIMIT])
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
                "max_margin_usd": max_order_amount_usd / roulette_info["leverage"],
            }
        return results_by_roulette_id

    def format_status(self) -> str:
        """
        Displays the three candlesticks involved in the script with RSI, BBANDS and EMA.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        pnl_usd_global = 0
        fees_cum_usd_global = 0
        lines = []
        total_results = ["########### BOT PERFORMACE ###########\n"]
        results_per_mkt = ["########### GAME HISTORY ###########\n"]
        active_roulette_per_mkt = ["########### ACTIVE ROULLETES ###########\n"]

        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            pnl_usd = 0
            fees_cum_usd = 0
            results_per_mkt.extend([f"\n--------> Trading Pair: {trading_pair}<--------\n"])
            active_roulette_per_mkt.extend([f"\n--------> Trading Pair: {trading_pair}<--------\n"])
            if roulette_info["candles"].is_ready:
                active_roulette_per_mkt.extend([])
                signal, take_profit, stop_loss, indicators = self.get_signal_tp_and_sl(roulette_info)
                active_roulette_per_mkt.extend([f"Market Data: Signal: {signal} | Take Profit: {take_profit:.4f} | Stop Loss: {stop_loss:.4f}\n"])
                active_roulette_per_mkt.extend([f"BB%: {indicators[0]:.2f} | MACDh: {indicators[1]:.6f} | MACD: {indicators[2]:.6f}"])
            else:
                active_roulette_per_mkt.extend(["", "  No data collected."])

            if len(roulette_info["active_executors"]) > 0:
                for executor in roulette_info["active_executors"]:
                    active_roulette_per_mkt.extend([f"|Roullete id: {executor.roulette_group_id} | Status: {executor.status} | Ball number: {executor.ball_number} |"])
                    active_roulette_per_mkt.extend(executor.to_format_status())
                    pnl_usd += executor.pnl_usd
                    fees_cum_usd += executor.cum_fees
                    pnl_usd_global += executor.pnl_usd
                    fees_cum_usd_global += executor.cum_fees

            if len(roulette_info["stored_executors"]) > 0:
                mkt_roullete_stats = self.calculate_roulette_stats(roulette_info).items()
                for executor in roulette_info["stored_executors"]:
                    pnl_usd_global += executor.pnl_usd
                    fees_cum_usd_global += executor.cum_fees
                    pnl_usd += executor.pnl_usd
                    fees_cum_usd += executor.cum_fees
                results_per_mkt.extend([
                f"\n| PNL: {pnl_usd:.4f} | Fees: {fees_cum_usd:.4f} | Net result: {(pnl_usd - fees_cum_usd):.4f}(USD)"])
                total_results.extend([
                f"\n {trading_pair} | Roullete Count: {len(mkt_roullete_stats)} | PNL: {pnl_usd:.4f} | Fees: {fees_cum_usd:.4f} | Net result: {(pnl_usd - fees_cum_usd):.4f}(USD)"])
                for roulette_id, stats in mkt_roullete_stats:
                    results_per_mkt.extend([f"""
- Roulette id: {roulette_id} | {stats['ball_numbers']} Balls |
| Realized PNL: {stats['realized_pnl']:.4f} | Fees: {stats['cum_fees']:.4f} | Net result: {stats['net_pnl']:.4f} |
| Max amount: {stats['max_order_amount_usd']:.4f} | Max margin: {stats['max_margin_usd']:.4f}
    """])
        total_results.insert(1,
            f"| PNL USD: {pnl_usd_global:.4f} | Fees cum USD: {fees_cum_usd_global:.4f} | Net result: {(pnl_usd - fees_cum_usd):.4f} |")
        lines.extend(total_results)
        lines.extend(results_per_mkt)
        lines.extend(active_roulette_per_mkt)
        return "\n".join(lines)

    def clean_and_store_executors(self):
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            current_price = self.connectors[self.exchange].get_mid_price(trading_pair)
            minutes = roulette_info["order_placed_time_limt"]
            waiting_executors = [x for x in roulette_info["active_executors"] if x.status == PositionExecutorStatus.ORDER_PLACED and x.position_config.timestamp + minutes*60 <= self.current_timestamp]
            for executor_waiting_to_enter in waiting_executors:
                open_order_entry_price = executor_waiting_to_enter._open_order._order.price
                buffer_used = 1 + roulette_info["limit_order_price_buffer"] if executor_waiting_to_enter.side == PositionSide.SHORT else 1 - roulette_info["limit_order_price_buffer"]
                price_when_order_placed = Decimal(open_order_entry_price)/Decimal(buffer_used)
                if abs(Decimal(current_price)-Decimal(open_order_entry_price)) > abs(Decimal(price_when_order_placed)-Decimal(open_order_entry_price)):
                    executor_waiting_to_enter.cancel_executor_order_placed()
            executors_to_store = [executor for executor in roulette_info["active_executors"] if executor.is_closed]
            if not os.path.exists(roulette_info["csv_path"]):
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
                df_header.to_csv(roulette_info["csv_path"], mode='a', header=False, index=False)
            for executor in executors_to_store:
                roulette_info["stored_executors"].append(executor)
                close_order_id = executor.close_order.order_id if executor.close_order else ""
                df = pd.DataFrame([(executor.timestamp,
                                    executor.roulette_group_id,
                                    executor.ball_number,
                                    executor.exchange,
                                    executor.trading_pair,
                                    executor.side,
                                    executor.amount,
                                    executor.pnl,
                                    executor.close_timestamp if close_order_id else '',
                                    executor.entry_price,
                                    executor.close_price,
                                    executor.status,
                                    executor.position_config.stop_loss,
                                    executor.position_config.take_profit,
                                    executor.position_config.time_limit,
                                    executor.open_order_type,
                                    roulette_info["leverage"],
                                    executor.open_order.order_id,
                                    close_order_id,
                                    executor.pnl_usd,
                                    executor.cum_fees
                                    )])
                df.to_csv(roulette_info["csv_path"], mode='a', header=False, index=False)
            roulette_info["active_executors"] = [executor for executor in roulette_info["active_executors"] if
                                                 not executor.is_closed]

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

    def is_margin_enough(self, betting_amount, trading_pair, roulette_info):
        quote_balance = self.connectors[self.exchange].get_balance(trading_pair.split("-")[-1])
        if betting_amount * Decimal("1.01") < quote_balance * Decimal(str(roulette_info["leverage"])):
            return True
        else:
            self.logger().info("No enough margin to place orders.")
            return False

    def check_and_set_leverage(self):
        if not self.set_leverage_flag:
            for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
                connector = self.connectors[self.exchange]
                connector.set_position_mode(PositionMode.HEDGE)
                connector.set_leverage(trading_pair=trading_pair, leverage=roulette_info["leverage"])
            self.set_leverage_flag = True

    def get_roulette_amount(self, trading_pair, roulette_info, take_profit):
        net_loss_usd = 0
        ball_number = 1
        failed_entries_in_roullete = 0
        for executor in reversed(roulette_info["stored_executors"]):
            if executor.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT:
                break
            else:
                if executor.status != PositionExecutorStatus.CANCELED_BY_TIME_LIMIT:
                    ball_number += 1
                    net_loss_usd += (executor.pnl_usd - executor.cum_fees)
                else:
                    failed_entries_in_roullete += 1
        extra_amount = - net_loss_usd / Decimal(
            take_profit - self.taker_fee - self.maker_fee) if net_loss_usd < Decimal(
            "0") else Decimal("0")
        if ball_number == 1 and failed_entries_in_roullete==0:
            roulette_info["roulette_group_id"] += 1
        if extra_amount == 0:
            ball_number = 1
        roulette_info["ball_number"] = ball_number
        amount = roulette_info["initial_order_amount_usd"] + extra_amount

        if not self.is_margin_enough(betting_amount=amount, trading_pair=trading_pair, roulette_info=roulette_info):
            self.notify_hb_app_with_timestamp(f"Game over! Not enough margin to bet {amount} for {trading_pair}")
            amount = roulette_info["initial_order_amount_usd"]
            roulette_info["roulette_group_id"] += 1
            roulette_info["ball_number"] = 1
        return amount

    def get_entry_price(self,trading_pair,roulette_info,position_side):

                    
        price_type = PriceType.BestAsk if position_side == PositionSide.SHORT else PriceType.BestBid
        price_limit_order = self.connectors[self.exchange].get_price_by_type(trading_pair, price_type)
        price_limit_buffer_multiplier = 1 + roulette_info["limit_order_price_buffer"] \
            if position_side == PositionSide.SHORT else 1 - roulette_info["limit_order_price_buffer"]
        return Decimal(price_limit_order) * Decimal(price_limit_buffer_multiplier)