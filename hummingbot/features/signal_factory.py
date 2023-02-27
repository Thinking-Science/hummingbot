import pandas as pd
import pandas_ta as ta  # noqa: F401


def trend_follower_with_bb_and_macd_filters(df: pd.DataFrame):
    df.ta.bbands(length=21, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df["std"] = df["close"].rolling(21).std()
    df["std_close"] = df["std"] / df["close"]
    last_row = df.iloc[-1]
    bbp = last_row["BBP_21_2.0"]
    m_21 = last_row["BBM_21_2.0"]
    macd = last_row["MACDh_12_26_9"]
    price = last_row["close"]
    take_profit = last_row["std_close"] * 2
    stop_loss = last_row["std_close"] * 1.25
    signal = 0
    if price > m_21 and bbp < 0.9 and macd > 0:
        signal = 1
    elif price < m_21 and bbp > 0.1 and macd < 0:
        signal = -1
    return df, signal, take_profit, stop_loss
