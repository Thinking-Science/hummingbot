import asyncio
import logging
from typing import Any, Dict, Optional

import numpy as np

from hummingbot.core.network_iterator import NetworkStatus, safe_ensure_future
from hummingbot.core.utils.tracking_nonce import get_tracking_nonce
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, WSJSONRequest
from hummingbot.core.web_assistant.ws_assistant import WSAssistant
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.data_feed.candles_feed.kucoin_spot_candles import constants as CONSTANTS
from hummingbot.logger import HummingbotLogger


class KucoinSpotCandles(CandlesBase):
    _logger: Optional[HummingbotLogger] = None
    _last_ws_message_sent_timestamp = 0
    _ping_interval = 0

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self, trading_pair: str, interval: str = "1min", max_records: int = 150):
        super().__init__(trading_pair, interval, max_records)

    @property
    def name(self):
        return f"kucoin_spot_{self._trading_pair}"

    @property
    def rest_url(self):
        return CONSTANTS.REST_URL

    @property
    def wss_url(self):
        return CONSTANTS.WSS_URL

    @property
    def health_check_url(self):
        return self.rest_url + CONSTANTS.HEALTH_CHECK_ENDPOINT

    @property
    def candles_url(self):
        return self.rest_url + CONSTANTS.CANDLES_ENDPOINT

    @property
    def rate_limits(self):
        return CONSTANTS.RATE_LIMITS

    @property
    def intervals(self):
        return CONSTANTS.INTERVALS

    async def check_network(self) -> NetworkStatus:
        rest_assistant = await self._api_factory.get_rest_assistant()
        await rest_assistant.execute_request(url=self.health_check_url,
                                             throttler_limit_id=CONSTANTS.HEALTH_CHECK_ENDPOINT)
        return NetworkStatus.CONNECTED

    def get_exchange_trading_pair(self, trading_pair):
        return trading_pair

    async def fetch_candles(self,
                            start_time: Optional[int] = None,
                            end_time: Optional[int] = None,
                            limit: Optional[int] = 500):
        rest_assistant = await self._api_factory.get_rest_assistant()
        params = {"symbol": self._ex_trading_pair, "interval": self.interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        candles = await rest_assistant.execute_request(url=self.candles_url,
                                                       throttler_limit_id=CONSTANTS.CANDLES_ENDPOINT,
                                                       params=params)

        return np.array(candles)[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]].astype(float)

    async def fill_historical_candles(self):
        max_request_needed = (self._candles.maxlen // 1000) + 1
        requests_executed = 0
        while not self.is_ready:
            missing_records = self._candles.maxlen - len(self._candles)
            end_timestamp = int(self._candles[0][0])
            try:
                if requests_executed < max_request_needed:
                    # we have to add one more since, the last row is not going to be included
                    candles = await self.fetch_candles(end_time=end_timestamp, limit=missing_records + 1)
                    # we are computing agaefin the quantity of records again since the websocket process is able to
                    # modify the deque and if we extend it, the new observations are going to be dropped.
                    missing_records = self._candles.maxlen - len(self._candles)
                    self._candles.extendleft(candles[-(missing_records + 1):-1][::-1])
                    requests_executed += 1
                else:
                    self.logger().error(f"There is no data available for the quantity of "
                                        f"candles requested for {self.name}.")
                    raise
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().exception(
                    "Unexpected error occurred when getting historical klines. Retrying in 1 seconds...",
                )
                await self._sleep(1.0)

    async def _subscribe_channels(self, ws: WSAssistant):
        """
        Subscribes to the candles events through the provided websocket connection.
        :param ws: the websocket assistant used to connect to the exchange
        """
        try:
            payload = {
                "id": str(get_tracking_nonce()),
                "type": "subscribe",
                "topic": f"/market/candles:{self._ex_trading_pair}_{self.interval}",
                "privateChannel": False,
                "response": False,
            }
            subscribe_candles_request: WSJSONRequest = WSJSONRequest(payload=payload)

            await ws.send(subscribe_candles_request)
            self.logger().info("Subscribed to public klines...")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().error(
                "Unexpected error occurred subscribing to public klines...",
                exc_info=True
            )
            raise

    async def _process_websocket_messages(self, websocket_assistant: WSAssistant):
        while True:
            try:
                seconds_until_next_ping = self._ping_interval - (self._time() - self._last_ws_message_sent_timestamp)
                await asyncio.wait_for(super()._process_websocket_messages_from_candles(websocket_assistant=websocket_assistant),
                                       timeout=seconds_until_next_ping)
            except asyncio.TimeoutError:
                payload = {
                    "id": str(get_tracking_nonce()),
                    "type": "ping",
                }
                ping_request = WSJSONRequest(payload=payload)
                self._last_ws_message_sent_timestamp = self._time()
                await websocket_assistant.send(request=ping_request)

    async def _process_websocket_messages_from_candles(self, websocket_assistant: WSAssistant):
        async for ws_response in websocket_assistant.iter_messages():
            data: Dict[str, Any] = ws_response.data
            if data is not None and data.get(
                    "e") == "kline":  # data will be None when the websocket is disconnected
                timestamp = data["k"]["t"]
                open = data["k"]["o"]
                high = data["k"]["h"]
                low = data["k"]["l"]
                close = data["k"]["c"]
                volume = data["k"]["v"]
                quote_asset_volume = data["k"]["q"]
                n_trades = data["k"]["n"]
                taker_buy_base_volume = data["k"]["V"]
                taker_buy_quote_volume = data["k"]["Q"]
                if len(self._candles) == 0:
                    self._candles.append(np.array([timestamp, open, high, low, close, volume,
                                                   quote_asset_volume, n_trades, taker_buy_base_volume,
                                                   taker_buy_quote_volume]))
                    safe_ensure_future(self.fill_historical_candles())
                elif timestamp > int(self._candles[-1][0]):
                    # TODO: validate also that the diff of timestamp == interval (issue with 1M interval).
                    self._candles.append(np.array([timestamp, open, high, low, close, volume,
                                                   quote_asset_volume, n_trades, taker_buy_base_volume,
                                                   taker_buy_quote_volume]))
                elif timestamp == int(self._candles[-1][0]):
                    self._candles.pop()
                    self._candles.append(np.array([timestamp, open, high, low, close, volume,
                                                   quote_asset_volume, n_trades, taker_buy_base_volume,
                                                   taker_buy_quote_volume]))

    async def _connected_websocket_assistant(self) -> WSAssistant:
        rest_assistant = await self._api_factory.get_rest_assistant()
        connection_info = await rest_assistant.execute_request(
            url=CONSTANTS.PUBLIC_WS_DATA_PATH_URL,
            method=RESTMethod.POST,
            throttler_limit_id=CONSTANTS.PUBLIC_WS_DATA_PATH_URL,
        )

        ws_url = connection_info["data"]["instanceServers"][0]["endpoint"]
        self._ping_interval = int(connection_info["data"]["instanceServers"][0]["pingInterval"]) * 0.8 * 1e-3
        token = connection_info["data"]["token"]

        ws: WSAssistant = await self._api_factory.get_ws_assistant()
        await ws.connect(ws_url=f"{ws_url}?token={token}", message_timeout=self._ping_interval)
        return ws
