import datetime
import os
from collections import deque
from decimal import Decimal
from typing import Deque, Dict, List

import pandas as pd
import pandas_ta as ta  # noqa: F401

from hummingbot import data_path
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide, TradeType
from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import OrderBookEvent, OrderBookTradeEvent
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.features.signal_factory import trend_follower_with_bb_and_macd_filters
from hummingbot.smart_components.position_executor.data_types import PositionConfig, PositionExecutorStatus
from hummingbot.smart_components.position_executor.position_executor import PositionExecutor
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class SignalExecutor(PositionExecutor):
    def __init__(self, position_config: PositionConfig, strategy: ScriptStrategyBase, signal_value: int,
                 executor_group_id: int = 1, ball_number: int = 1):
        super().__init__(position_config, strategy)
        self.signal_value = signal_value
        self.executor_group_id = executor_group_id
        self.ball_number = ball_number


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
    stored_executors: Deque[SignalExecutor] = deque(maxlen=20)  # Store only the last 20 executors for reporting

    # Configure the parameters for the position
    taker_fee = 0.0003
    maker_fee = 0.00012
    total_fee = taker_fee * 2
    min_stop_loss = 0.002
    time_limit = 60 * 10  # 10 minutes

    short_threshold = -0.5
    long_threshold = 0.5

    position_side = PositionSide.LONG
    signal_side = PositionSide.LONG

    # Create the candles that we want to use and the thresholds for the indicators
    candles_1m = CandlesFactory.get_candle(connector=exchange,
                                           trading_pair=trading_pair,
                                           interval="1m", max_records=500)
    # candles_3m = CandlesFactory.get_candle(connector=exchange,
    #                                        trading_pair=trading_pair,
    #                                        interval="3m", max_records=500)
    candles = {
        f"{trading_pair}_1m": candles_1m,
        # f"{trading_pair}_3m": candles_3m,
    }

    # Configure the leverage and order amount the bot is going to use
    set_leverage_flag = None
    leverage = 20
    initial_order_amount_usd = Decimal("6")
    executor_group_id = 1
    ball_number = 1

    today = datetime.datetime.today()
    csv_path = data_path() + f"/roulette_{exchange}_{trading_pair}_{today.day:02d}-{today.month:02d}-{today.year}.csv"

    # Suscribe to trades
    subscribed_to_order_book_trade_event: bool = False
    trades_buffer = deque(maxlen=20000)
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
        if not self.subscribed_to_order_book_trade_event:
            self.subscribe_to_order_book_trade_event()
        self.check_and_set_leverage()
        if len(self.get_active_executors()) < self.max_executors and self.all_candles_ready:
            signal, take_profit, stop_loss = self.get_signal()
            if (signal < self.short_threshold or signal > self.long_threshold) and stop_loss > self.min_stop_loss:
                self.logger().info(f"Signal: {signal:.2f}, take_profit: {take_profit:.2f}, stop_loss: {stop_loss:.2f}")
                bet = self.get_roulette_amount(take_profit)
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
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        time_limit=self.time_limit),
                    strategy=self,
                    signal_value=signal,
                    executor_group_id=self.executor_group_id,
                    ball_number=self.ball_number
                )
                self.active_executors.append(signal_executor)
        self.clean_and_store_executors()

    def subscribe_to_order_book_trade_event(self):
        self.order_book_trade_event = SourceInfoEventForwarder(self._process_public_trade)
        for market in self.connectors.values():
            for order_book in market.order_books.values():
                order_book.add_listener(OrderBookEvent.TradeEvent, self.order_book_trade_event)
        self.subscribed_to_order_book_trade_event = True

    def _process_public_trade(self, event_tag: int, market: ConnectorBase, event: OrderBookTradeEvent):
        self.trades_buffer.append(event)

    def get_signal(self):
        values = []
        for candle_name, candle in self.candles.items():
            candles_df = candle.candles_df
            candles, signal, take_profit, stop_loss = trend_follower_with_bb_and_macd_filters(candles_df)
            values.append(signal)
        return values[0], take_profit, stop_loss

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
        pnl_usd = 0
        fees_cum_usd = 0

        if len(self.stored_executors) > 0:
            lines.extend([
                "\n########################################## Closed Executors ##########################################"])

        for executor in self.stored_executors:
            lines.extend([f"|Signal id: {executor.timestamp} | Signal value: {executor.signal_value:.2f} | Ball number: {executor.ball_number} |"])
            lines.extend(executor.to_format_status())
            lines.extend([
                "-----------------------------------------------------------------------------------------------------------"])
            pnl_usd += executor.pnl_usd
            fees_cum_usd += executor.cum_fees
        if len(self.active_executors) > 0:
            lines.extend([
                "\n########################################## Active Executors ##########################################"])
            for executor in self.active_executors:
                lines.extend([f"|Signal id: {executor.timestamp} | Signal value: {executor.signal_value:.2f} | Ball number: {executor.ball_number} |"])
                lines.extend(executor.to_format_status())
                pnl_usd += executor.pnl_usd
                fees_cum_usd += executor.cum_fees
                if executor.take_profit_order.order:
                    volume_to_take_profit = self.connectors[executor.exchange].get_volume_for_price(executor.trading_pair, is_buy=True if executor.side == PositionSide.LONG else False, price=executor.take_profit_price)
                    volume_to_stop_loss = self.connectors[executor.exchange].get_volume_for_price(executor.trading_pair, is_buy=False if executor.side == PositionSide.LONG else True, price=executor.stop_loss_price)
                    lines.extend([f"| Stop loss volume: {volume_to_stop_loss.result_volume:.4f} | Take profit volume: {volume_to_take_profit.result_volume:.4f} "])

        lines.extend([f"\n| PNL USD: {pnl_usd:.4f} | Fees cum USD: {fees_cum_usd:.4f} | Net result: {(pnl_usd - fees_cum_usd):.4f} |"])

        if self.all_candles_ready:
            lines.extend([
                "\n############################################ Market Data ############################################\n"])
            # trades_to_filter = [25, 50, 100, 250, 500, 1000, 5000, 10000, 20000]
            # for n_trades in trades_to_filter:
            #     if len(self.trades_buffer) > n_trades:
            #         buy_volume, sell_volume, net_volume = self.get_volume_of_last_trades(n_trades)
            #         lines.extend([f"""| Sell:{sell_volume:.1f} | Buy: {buy_volume:.1f} | Net: {net_volume:.1f} | Trades: {n_trades}| """])
            # buy_volume, sell_volume, net_volume = self.get_volume_of_last_trades(len(self.trades_buffer))
            # lines.extend([f"| Sell:{sell_volume:.1f} | Buy: {buy_volume:.1f} | Total: {net_volume:.1f} | Trades {len(self.trades_buffer)} "])
            columns_to_show = ["timestamp", "open", "low", "high", "close", "volume", "BBP_21_2.0", "BBM_21_2.0", "MACDh_5_21_9", "MACD_5_21_9"]
            for candle_name, candles in self.candles.items():
                candles_df = candles.candles_df
                # Let's add some technical indicators
                candles_df, signal, take_profit, stop_loss = trend_follower_with_bb_and_macd_filters(candles_df)
                candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], unit="ms")
                lines.extend([f"Candles: {candles.name} | Interval: {candles.interval}\n"])
                candles_df["std"] = candles_df["close"].rolling(21).std()
                candles_df["std_close"] = candles_df["std"] / candles_df["close"]
                candles_df["stop_loss"] = 0.75 * candles_df["std_close"]
                candles_df["take_profit"] = 1.75 * candles_df["std_close"]
                lines.extend(
                    ["    " + line for line in
                     candles_df[columns_to_show].tail().to_string(index=False).split("\n")])
                # Signal Genaration
                lines.extend([f"Signal: {signal}"])
            lines.extend(["\n-----------------------------------------------------------------------------------------------------------\n"])
        else:
            lines.extend(["", "  No data collected."])
        return "\n".join(lines)

    def get_volume_of_last_trades(self, n_trades: int):
        last_n_trades = list(self.trades_buffer)[-n_trades:]
        buy_amount_n_trades = 0
        sell_amount_n_trades = 0
        for trade in last_n_trades:
            if trade.type == TradeType.BUY:
                buy_amount_n_trades += trade.amount * trade.price
            elif trade.type == TradeType.SELL:
                sell_amount_n_trades += trade.amount * trade.price
        net_volume = buy_amount_n_trades - sell_amount_n_trades
        return buy_amount_n_trades, sell_amount_n_trades, net_volume

    def clean_and_store_executors(self):
        executors_to_store = [executor for executor in self.active_executors if executor.is_closed]
        if not os.path.exists(self.csv_path):
            df_header = pd.DataFrame([("timestamp",
                                       "executor_group_id",
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
            if executor.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT:
                close_order_id = executor.take_profit_order.order_id
            elif executor.status == PositionExecutorStatus.CLOSED_BY_STOP_LOSS:
                close_order_id = executor.stop_loss_order.order_id
            elif executor.status == PositionExecutorStatus.CLOSED_BY_TIME_LIMIT:
                close_order_id = executor.time_limit_order.order_id
            else:
                close_order_id = ""
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

        if ball_number == 1:
            self.executor_group_id += 1
        self.ball_number = ball_number
        extra_amount = - net_loss_usd / Decimal(take_profit - self.taker_fee - self.maker_fee) if net_loss_usd < Decimal("0") else Decimal("0")
        amount = self.initial_order_amount_usd + extra_amount
        self.logger().info(f"Ball number: {ball_number}")
        self.logger().info(f'Net_loss_usd: {net_loss_usd}')
        self.logger().info(f'Amount based in net_loss_usd: {amount}')
        return amount
