import os, pickle

from tensortrade.data.cdd import CryptoDataDownload
import ta
import numpy as np
from enum import Enum


class STATE(Enum):
    TRAIN = 1
    EVAL = 2
    TEST = 3


class DataLoader:

    def add_technical_indicators(self):
        ta.add_all_ta_features(
            self.btc_historical_data,
            fillna=True,
            colprefix=self.prefix,
            **{k: self.prefix + k for k in ['open', 'high', 'low', 'close', 'volume']}
        )
        return self.btc_historical_data

    def __init__(self, pkl_name, crypto_to_trade, fraction_of_data_to_use=1, prefix=None):

        self.prefix = crypto_to_trade + ":" if not prefix else prefix

        file = os.path.join("sam_env/data", pkl_name) + ".pkl"

        if os.path.exists(file):
            with open(file, "rb") as pkl:
                self.btc_historical_data = pickle.load(pkl)
        else:
            cdd = CryptoDataDownload()
            self.btc_historical_data = cdd.fetch("Bitfinex", "USD", crypto_to_trade, "1h").add_prefix(self.prefix)
            self.btc_historical_data = self.add_technical_indicators()
            self. btc_historical_data.drop(columns=[self.prefix + "unix", self.prefix + "date"], inplace=True)
            self.btc_historical_data.bfill(inplace=True)
            self.btc_historical_data.replace([np.inf, -np.inf], 0, inplace=True)

            with open(file, "wb") as pkl:
                pickle.dump(self.btc_historical_data, pkl)

    def __call__(self, state: STATE):

        total_length = len(self.btc_historical_data)
        train_size = int(0.70 * total_length)
        other_size = int(0.15 * total_length)
        if state == STATE.TRAIN:
            return self.btc_historical_data[:train_size]
        elif state == STATE.EVAL:
            return self.btc_historical_data[train_size: train_size + other_size]
        else:
            return self.btc_historical_data[-other_size:]
