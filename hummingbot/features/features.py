from itertools import chain
from math import pi

import numpy as np
import pandas as pd
import pandas_ta as ta


class Features:
    @staticmethod
    def intersection(
            x: pd.Series,
            y1: pd.Series,
            y2: pd.Series,
            y1_breakup: str,
            y2_breakup: str
    ):
        # If macd is greater than signal 1, else 0
        cross_signal = pd.Series(np.where((y1 == np.nan) | (y2 == np.nan),
                                          0,
                                          np.where(y1 > y2, 1.0, 0.0)))
        # If last macd_cross_signal != current signal then it's a cross
        cross_event = pd.Series(np.where(cross_signal != cross_signal.shift(1), 1, 0))
        cross_side = pd.Series(np.where((y1 > y2) & (cross_event > 0), y1_breakup,
                                        np.where((y1 < y2) & (cross_event > 0), y2_breakup, "BOTH")))

        x = pd.to_datetime(x)

        x_prev = x.shift()
        y1_prev = y1.shift()
        y2_prev = y2.shift()

        df = pd.concat([x, y1, y2, x_prev, y1_prev, y2_prev, cross_event, cross_side], axis=1)
        df.columns = ['x', 'y1', 'y2', 'x_prev', 'y1_prev', 'y2_prev', 'cross_event', 'cross_side']

        df['tan_1'] = np.where(df['cross_event'] > 0,
                               (df['y1'] - df['y1_prev']) / (df['x'] - df['x_prev']).dt.total_seconds().div(60), 0)
        df['alpha_1'] = np.arctan(df['tan_1'])

        df['tan_2'] = np.where(df['cross_event'] > 0,
                               (df['y2'] - df['y2_prev']) / (df['x'] - df['x_prev']).dt.total_seconds().div(60), 0)
        df['alpha_2'] = np.arctan(df['tan_2'])

        df['alpha'] = np.where(
            (df['alpha_1'] * df['alpha_2'] < 0),
            abs(df['alpha_1']) + abs(df['alpha_2']),
            np.where(
                df['alpha_1'] * df['alpha_2'] > 0,
                abs(df[['alpha_1', 'alpha_2']]).max(axis=1) - abs(df[['alpha_1', 'alpha_2']]).min(axis=1), 0))
        #####
        df['value'] = np.where(
            df['cross_side'] == "SHORT",
            -df['alpha'] / pi,
            np.where(
                df['cross_side'] == "LONG",
                df['alpha'] / pi,
                0))
        df.replace({'LONG': 1, 'SHORT': -1, 'BOTH': 0}, inplace=True)
        return df['cross_event'], df['alpha'], df['value'], df['cross_side']

    @staticmethod
    def crossover(
            base: pd.Series,
            upper_thold: pd.Series,
            upper_side: str,
            down_thold: pd.Series,
            down_side: str):

        df = pd.concat([base, upper_thold, down_thold], axis=1)
        df.columns = ['base', 'upper_thold', 'down_thold']
        df['side'] = np.where(df['base'] < df['down_thold'],
                              down_side,
                              np.where(df['base'] > df['upper_thold'],
                                       upper_side,
                                       "BOTH"))

        df['memory'] = df.groupby((df['side'] != df['side'].shift()).cumsum())['side'].cumcount()
        df.replace({'LONG': 1, 'SHORT': -1, 'BOTH': 0}, inplace=True)
        return df['side'], df['memory']

    def add_features(self, df: pd.DataFrame, features_dict, dropna=False):
        features = {
            feature: [
                [dict(config['config'], **{'kind': indicator_name}) for config in configs]
                for indicator_name, configs in indicator.items()]
            for feature, indicator in features_dict.items()}

        features_crossover = [feature for group in features['crossover'] for feature in group]
        features_intersection = [feature for group in features['intersection'] for feature in group]
        all_features = features_crossover + features_intersection
        ta_strategy = ta.Strategy(
            name="RMB",
            ta=all_features
        )

        df.ta.strategy(ta_strategy)

        add_features_dict = list()
        add_features_dict.append([[dict(feature, **{'kind': indicator_name, 'indicator_type': indicator_type}) for
                                   indicator_name, features in
                                   features_dict[indicator_type].items() for feature in features] for indicator_type in
                                  features_dict.keys()])

        for feature in list(chain.from_iterable(add_features_dict[0])):
            # Build base and tholds names
            if feature['indicator_type'] == 'crossover':
                # rsi
                if feature['kind'] == 'rsi':
                    feature['name'] = feature['base'] = 'RSI_{}'.format(feature['config']['length'])
                # bbands
                if feature['kind'] == 'bbands':
                    feature['base'] = 'close'
                    std = feature['config'].get('std', '2.0')
                    feature['name'] = 'BB_{}_{}'.format(feature['config']['length'], std)
                    feature['upper_thold'] = 'BBU_{}_{}'.format(feature['config']['length'], std)
                    feature['down_thold'] = 'BBL_{}_{}'.format(feature['config']['length'], std)

                # Run crossover and append to df
                memory = feature['name'] + '_MEMO'
                side = feature['name'] + '_SIDE'

                if isinstance(feature['upper_thold'], int):
                    upper_thold = pd.Series(feature['upper_thold'], index=np.arange(len(df)))
                elif isinstance(feature['upper_thold'], str):
                    upper_thold = df[feature['upper_thold']]

                if isinstance(feature['down_thold'], int):
                    down_thold = pd.Series(feature['down_thold'], index=np.arange(len(df)))
                elif isinstance(feature['down_thold'], str):
                    down_thold = df[feature['down_thold']]

                df[side], df[memory] = self.crossover(base=df[feature['base']],
                                                      upper_thold=upper_thold,
                                                      upper_side=feature['upper_side'],
                                                      down_thold=down_thold,
                                                      down_side=feature['down_side']
                                                      )

            elif feature['indicator_type'] == 'intersection':
                # macd
                if feature['kind'] == 'macd':
                    feature['series1'] = feature['name'] = 'MACD_{}_{}_{}'.format(feature['config']['fast'],
                                                                                  feature['config']['slow'],
                                                                                  feature['config']['signal'])
                    feature['series2'] = 'MACDs_{}_{}_{}'.format(feature['config']['fast'], feature['config']['slow'],
                                                                 feature['config']['signal'])

                # Run crossover and append to df
                alpha = feature['name'] + '_ALPHA'
                side = feature['name'] + '_SIDE'
                value = feature['name'] + '_VALUE'
                cross = feature['name'] + '_CROSS'
                df[cross], df[alpha], df[value], df[side] = self.intersection(df['timestamp'], df[feature['series1']],
                                                                              df[feature['series2']],
                                                                              feature['series1_breakup_side'],
                                                                              feature['series2_breakup_side'])
        if dropna:
            return df.dropna()
        else:
            return df
