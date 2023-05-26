from bidict import bidict

from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit

REST_URL = "https://api.kucoin.com"
HEALTH_CHECK_ENDPOINT = "/api/v3/ping"
CANDLES_ENDPOINT = "/api/v3/klines"

PUBLIC_WS_DATA_PATH_URL = "/api/v1/bullet-public"
WSS_URL = "wss://stream.binance.com:9443/ws"

INTERVALS = bidict({
    "1s": 1,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000
})

REQUEST_WEIGHT = "REQUEST_WEIGHT"

RATE_LIMITS = [
    RateLimit(REQUEST_WEIGHT, limit=1200, time_interval=60),
    RateLimit(CANDLES_ENDPOINT, limit=1200, time_interval=60, linked_limits=[LinkedLimitWeightPair("raw", 1)]),
    RateLimit(HEALTH_CHECK_ENDPOINT, limit=1200, time_interval=60, linked_limits=[LinkedLimitWeightPair("raw", 1)])]
