import datetime
import os
import time
from decimal import Decimal
from typing import Dict

import pandas as pd
import pandas_ta as ta  # noqa: F401

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.smart_components.position_executor.data_types import PositionExecutorStatus
from hummingbot.smart_components.roulette.roulette import Roulette, RouletteConfig, RouletteStatus
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


def get_leverage_by_trading_pair(trading_pair: str) -> int:
    trading_pair = 20
    return trading_pair


class LooterKing(ScriptStrategyBase):
    """
    A simple trading strategy that uses RSI in one timeframe to determine whether to go long or short.
    IMPORTANT: Binance perpetual has to be in Single Asset Mode, soon we are going to support Multi Asset Mode.
    """
    # Define the trading pair and exchange that we want to use and the csv where we are going to store the entries
    trading_pairs = ["DODO-BUSD","XRP-BUSD","FTM-BUSD","GALA-BUSD","TRX-BUSD","DOGE-BUSD","AGIX-BUSD"]
    exchange = "binance_perpetual"

    # Fee structure
    taker_fee = Decimal("0.0003")
    maker_fee = Decimal("0.00012")
    total_fee = taker_fee * 2
    today = datetime.datetime.today()

    # Maximum position executors at a time
    max_roulettes = 1
    roulette_by_trading_pair = {}
    for trading_pair in trading_pairs:
        roulette_by_trading_pair[trading_pair] = dict(roulette_config=RouletteConfig(
            exchange=exchange,
            trading_pair=trading_pair,
            stop_loss_multiplier=Decimal("0.5"),
            take_profit_multiplier=Decimal("1.0"),
            time_limit=60 * 30,
            max_stop_loss=Decimal("0.002"),
            trailing_stop_loss=False,
            trailing_stop_loss_pct=Decimal("0"),
            open_order_type=OrderType.LIMIT,
            open_order_refresh_analyze_time=30,
            open_order_buffer_price=Decimal("0.00001"),
            max_balls=7,
            initial_order_amount=Decimal("10.0"),
            leverage=get_leverage_by_trading_pair(trading_pair=trading_pair)),
            active_roulettes=[],
            stored_roulettes=[],
            candles_3m=CandlesFactory.get_candle(connector=exchange,
                                                 trading_pair=trading_pair,
                                                 interval="3m", max_records=500),
            candles_1m=CandlesFactory.get_candle(connector=exchange,
                                                 trading_pair=trading_pair,
                                                 interval="1m", max_records=500),
            trading_cash_out_time=0.25,  # days
            cashing_out=False,
            active_trading=True,
            max_game_overs=3,
            csv_path=data_path() + f"/roulette_{exchange}_{trading_pair}_{today.day:02d}-{today.month:02d}-{today.year}-{today.hour}.csv",
        )
    short_threshold = -0.5
    long_threshold = 0.5

    # Configure the leverage and order amount the bot is going to use
    set_leverage_flag = None

    markets = {exchange: set(trading_pairs)}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        # Is necessary to start the Candles Feed.
        super().__init__(connectors)
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            roulette_info["candles_3m"].start()
            roulette_info["candles_1m"].start()
        self.start_time = time.time()
        # TODO: make cashout property of the strategy

    def on_stop(self):
        """
        Without this functionality, the network iterator will continue running forever after stopping the strategy
        That's why is necessary to introduce this new feature to make a custom stop with the strategy.
        """
        # we are going to close all the open positions when the bot stops
        self.close_open_positions()
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            for active_roulette in roulette_info["active_roulettes"]:
                active_roulette.status = RouletteStatus.CLOSED_BY_COMMAND
            roulette_info["candles_3m"].stop()
            roulette_info["candles_1m"].stop()

    def on_tick(self):
        self.clean_and_store_roulettes()  # Stores closed position and refresh orders waiting to enter
        self.check_and_set_leverage()
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            self.set_cash_out_trigger(trading_pair, roulette_info)
            self.check_active_trading(roulette_info)
            if len(roulette_info["active_roulettes"]) < self.max_roulettes and \
                    roulette_info["candles_3m"].is_ready and \
                    roulette_info["candles_1m"].is_ready and \
                    roulette_info["active_trading"] and \
                    not roulette_info["cashing_out"]:
                self.logger().info(f"Creating new Roulette for {trading_pair}...")
                roulette = Roulette(
                    strategy=self,
                    roulette_config=roulette_info["roulette_config"],
                )
                roulette_info["active_roulettes"].append(roulette)

    def set_cash_out_trigger(self, trading_pair, roulette_info):
        life_seconds = roulette_info['trading_cash_out_time'] * 24 * 60 * 60
        if self.current_timestamp - self.start_time >= life_seconds:
            if not roulette_info['cashing_out']:
                self.logger().info(f"Cashing out for {trading_pair}!")
                roulette_info['cashing_out'] = True

    def get_signal_std(self, trading_pair):
        candles = self.roulette_by_trading_pair[trading_pair]["candles_3m"]
        candles_df = candles.candles_df
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
        candles_1m = self.roulette_by_trading_pair[trading_pair]["candles_1m"]
        candles_1m_df = candles_1m.candles_df
        candles_1m_df.ta.bbands(length=100, append=True)
        candles_1m_df.ta.macd(fast=21, slow=42, signal=9, append=True)
        candles_1m_df["std"] = candles_1m_df["close"].rolling(100).std()
        candles_1m_df["mean"] = candles_1m_df["close"].rolling(100).mean()
        candles_1m_df["std_close"] = candles_1m_df["std"] / candles_1m_df["close"]
        std_mean = candles_1m_df["std_close"].mean()
        last_candle = candles_1m_df.iloc[-1]
        bbp = last_candle["BBP_100_2.0"]
        max_bbands_width = (candles_1m_df['BBU_100_2.0'] - candles_1m_df['BBL_100_2.0']).max()
        actual_bbands_width = (last_candle['BBU_100_2.0'] - last_candle['BBL_100_2.0']).mean()
        bbands_perc = actual_bbands_width / max_bbands_width
        bbp_1m = last_candle["BBP_100_2.0"]
        macdh_1m = last_candle["MACDh_21_42_9"]
        macd_1m = last_candle["MACD_21_42_9"]
        std_pct_1m = last_candle["std_close"]

        if bbp < 0.25 and macdh_1m > 0 and bbands_perc >= 0.5:
            signal_value = 1
        elif bbp > 0.75 and macdh_1m < 0 and bbands_perc >= 0.5:
            signal_value = -1
        else:
            signal_value = 0
        return signal_value, Decimal(str(min(std_pct_1m, std_pct))), Decimal(str(std_mean))

    def format_status(self) -> str:
        """
        Displays the three candlesticks involved in the script with RSI, BBANDS and EMA.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        realized_net_pnl_global = Decimal(0)
        unrealized_net_pnl_global = Decimal(0)
        runaway_global = Decimal(0)
        active_roulettes_resume = ''
        active_roulettes_global = 0
        balls_paying_global = 0
        balls_waiting_enter_global = 0
        early_loot_roulettes_global = 0
        loot_roulettes_global = 0
        game_over_roulettes_global = 0
        lines = []
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            realized_net_pnl = Decimal(0)
            unrealized_net_pnl = Decimal(0)
            closed_by_early_loot = 0
            closed_by_stop_loss = 0
            closed_by_time_limit = 0
            closed_by_loot = 0
            canceled_by_time_limit = 0
            early_loot_roulettes = 0
            loot_roulettes = 0
            game_over_roulettes = 0
            roulette_history_resume = ''
            all_roulettes = roulette_info["stored_roulettes"] + roulette_info["active_roulettes"]
            for roulette in all_roulettes:
                unrealized_net_pnl += roulette.unrealized_net_pnl
                realized_net_pnl += roulette.realized_net_pnl
                if roulette.status == RouletteStatus.CLOSED_BY_GAME_OVER:
                    game_over_roulettes += 1
                elif roulette.status == RouletteStatus.CLOSED_BY_EARLY_LOOT:
                    early_loot_roulettes += 1
                elif roulette.status == RouletteStatus.CLOSED_BY_EARLY_LOOT:
                    loot_roulettes += 1
                closed_by_early_loot += roulette.closed_by_early_loot
                closed_by_stop_loss += roulette.closed_by_stop_loss
                closed_by_time_limit += roulette.closed_by_time_limit
                canceled_by_time_limit += roulette.canceled_by_time_limit
                closed_by_loot += roulette.closed_by_loot
                roulette_history_resume += f"|{(roulette.realized_net_pnl + roulette.unrealized_net_pnl):.4f}B{roulette.ball_number}|"
            unrealized_net_pnl_global += unrealized_net_pnl
            realized_net_pnl_global += realized_net_pnl
            early_loot_roulettes_global += early_loot_roulettes
            loot_roulettes_global += loot_roulettes
            game_over_roulettes_global += game_over_roulettes

            if roulette_info['active_trading']:
                if roulette_info['cashing_out']:
                    mkt_activity_status = "CASHING OUT"
                else:
                    mkt_activity_status = "ACTIVE"
            else:
                mkt_activity_status = "DONE"

            roulete_status_report = "ZZZ"
            game_over_usd = 0
            stop_loss = 0
            max_stop_loss = 0
            take_profit = 0
            std = 0
            signal = 0
            runaway = Decimal("0")
            if len(roulette_info["active_roulettes"]) > 0:
                current_roulette = roulette_info["active_roulettes"][-1]
                signal = current_roulette.signal
                game_over_usd = current_roulette.game_over_usd
                max_stop_loss = current_roulette._roulette_config.max_stop_loss
                stop_loss = current_roulette.stop_loss
                take_profit = current_roulette.take_profit
                std = current_roulette.std_mean
                runaway = current_roulette.realized_net_pnl + current_roulette.unrealized_net_pnl
                if len(current_roulette.executors) > 0:
                    active_roulettes_global += 1
                    current_roulette_status = current_roulette.executors[-1].status
                    if len([executor for executor in current_roulette.executors if executor.status != PositionExecutorStatus.CANCELED_BY_TIME_LIMIT]) > 0:
                        active_roulettes_resume += f"|{trading_pair.split('-')[0]}{(current_roulette.realized_net_pnl + current_roulette.unrealized_net_pnl):.4f}B{current_roulette.ball_number}|"
                    if current_roulette_status == PositionExecutorStatus.ORDER_PLACED:
                        roulete_status_report = "ORDER PLACED"
                        balls_waiting_enter_global += 1
                    elif current_roulette_status == PositionExecutorStatus.ACTIVE_POSITION:
                        roulete_status_report = "PLAYING"
                        balls_paying_global += 1
                        current_price = self.connectors[self.exchange].get_mid_price(trading_pair)
                        runaway += - self.taker_fee*current_roulette.executors[-1].amount*Decimal(current_price)
                    roulete_status_report += "|" + str(current_roulette.ball_number)
                runaway_global += runaway
            lines.extend([f"""
|{trading_pair}| {mkt_activity_status} | {roulete_status_report} |
|Net realized PNL: {realized_net_pnl:.4f} USD
|Net unrealized PNL: {unrealized_net_pnl:.4f}
|Game over USD: {game_over_usd:.4f}
|Runaway: {runaway:.4f}
|Signal: {signal}
|Stop Loss: {stop_loss:.4f}
|Take Profit: {take_profit:.4f}
|STD Mean: {std:.4f}
|Loots: {closed_by_loot}
|Early loots: {closed_by_early_loot}
|Stop Loss: {closed_by_stop_loss}
|Time Limit: {closed_by_time_limit}
|Expired: {canceled_by_time_limit}
|--*ROULETTE HISTORY*--
|WON{loot_roulettes}|EL{early_loot_roulettes}|GO{game_over_roulettes} 
{roulette_history_resume}
"""])
            info_by_closed_roulette = []
            for roulette in roulette_info["stored_roulettes"]:
                total_balls = roulette.closed_by_loot + roulette.closed_by_early_loot + roulette.closed_by_stop_loss + roulette.closed_by_time_limit
                info_by_closed_roulette.append(f"{total_balls}B{roulette.realized_net_pnl}")
            lines.extend(" | ".join(info_by_closed_roulette))
        lines.extend(["\n\n### BOT PERFORMACE ###"])
        lines.extend([f"""
Net realized PNL: {realized_net_pnl_global:.4f} USD
Net unrealized PNL: {unrealized_net_pnl_global:.4f} USD
Runaway: {runaway_global:.4f} USD
--*ROULETTES HISTORY*--
WON{loot_roulettes_global}|EL{early_loot_roulettes_global}|GO{game_over_roulettes_global} 
--*ACTIVE ROULETTES: {active_roulettes_global} *--
{active_roulettes_resume}
--*ACTIVE BALLS*--
Playing: {balls_paying_global} - Order placed:{balls_waiting_enter_global}
"""])
        return "\n".join(lines)

    def clean_and_store_roulettes(self):
        for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
            roulettes_to_store = [roulette for roulette in roulette_info["active_roulettes"] if roulette.is_closed()]
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
            for roulette in roulettes_to_store:
                roulette_info["stored_roulettes"].append(roulette)
                for executor in roulette.executors:
                    close_order_id = executor.close_order.order_id if executor.close_order else ""
                    df = pd.DataFrame([(executor.timestamp,
                                        roulette.roulette_id,
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
                                        roulette._roulette_config.leverage,
                                        executor.open_order.order_id,
                                        close_order_id,
                                        executor.pnl_usd,
                                        executor.cum_fees
                                        )])
                    df.to_csv(roulette_info["csv_path"], mode='a', header=False, index=False)
            roulette_info["active_roulettes"] = [roulette for roulette in roulette_info["active_roulettes"] if
                                                 not roulette.is_closed()]

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

    def check_and_set_leverage(self):
        if not self.set_leverage_flag:
            for trading_pair, roulette_info in self.roulette_by_trading_pair.items():
                connector = self.connectors[self.exchange]
                connector.set_position_mode(PositionMode.HEDGE)
                connector.set_leverage(trading_pair=trading_pair, leverage=roulette_info["roulette_config"].leverage)
            self.set_leverage_flag = True

    def check_active_trading(self, roulette_info):
        if len(roulette_info["active_roulettes"]) == 0:
            if roulette_info["cashing_out"]:
                roulette_info["active_trading"] = False
            cum_game_overs = 0
            for roulette in roulette_info["stored_roulettes"]:
                if roulette.status == RouletteStatus.CLOSED_BY_GAME_OVER:
                    cum_game_overs += 1
            if cum_game_overs >= roulette_info["max_game_overs"]:
                roulette_info["active_trading"] = False
