import datetime
import os
import pickle
import random
import multiprocessing as mp
import gym
import numpy as np
import pandas as pd
import ta
import tensortrade.env.default as default
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.env.default import actions, rewards, observers, stoppers, renderers, informers
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
from tensortrade.env.generic import TradingEnv
from tensortrade.env.generic.components.renderer import AggregateRenderer
from tensortrade.feed import NameSpace
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

from sam_env.data_prep import DataLoader, STATE


class SamSingle(gym.ObservationWrapper):

    def __init__(self,
                 state: STATE,
                 data_pickle_name="1h_data",
                 crypto_to_trade='BTC',
                 stop_loss=0.3,
                 window_size=1,
                 start_qty=0,
                 start_capital=10000,
                 end_index=None):

        self.crypto_to_trade = crypto_to_trade
        self.window_size = window_size
        self.stop_loss = stop_loss
        self.crypto_to_trade = crypto_to_trade
        self.pkl_name = data_pickle_name
        self.end = end_index
        self.state = state

        if isinstance(start_capital, float) or isinstance(start_capital, int):
            self.start_capital = start_capital
        elif isinstance(start_capital, tuple):
            if len(start_capital) == 1:
                self.start_capital = start_capital[0]
            elif len(start_capital) == 2:
                self.start_capital = random.randrange(start_capital[0], start_capital[1])
            else:
                raise TypeError(" tuple value for start capital has contain only 1 or 2 numbers")
        else:
            raise TypeError("start capital cant be type ", type(start_capital), " only tuple|float|int")

        self.start_qty = start_qty
        env = self.create_single_tick_env()

        super().__init__(env)

        self.full = None
        self.observation_space.shape = (self.observation_space.shape[-1],)

    @staticmethod
    def run(data, prefix):
        ta.add_all_ta_features(
            data,
            fillna=True,
            colprefix=prefix,
            **{k: prefix + k for k in ['open', 'high', 'low', 'close', 'volume']}
        )
        return data

    def create_data_stream_and_feed(self):

        btc_historical_data = DataLoader(self.pkl_name, self.crypto_to_trade)(self.state)

        btc_historical_data = btc_historical_data.iloc[:]

        with NameSpace("Bitfinex"):
            bitfinex_btc_streams = [
                Stream.source(list(btc_historical_data[c]), dtype="float").rename(c)
                for c in btc_historical_data.columns]

        data_feed = DataFeed(bitfinex_btc_streams)
        data_feed.compile()

        return data_feed, btc_historical_data

    def create_single_tick_env(self):

        data_feed, btc_historical_data = self.create_data_stream_and_feed()
        close_prices = btc_historical_data[self.crypto_to_trade + ":close"]

        # simulated exchange
        bitfinex = Exchange("bitfinex", service=execute_order)(
            Stream.source(list(close_prices), dtype="float").rename("USD-" + self.crypto_to_trade))

        cash = Wallet(bitfinex, self.start_capital * USD)
        asset = Wallet(bitfinex, self.start_qty * BTC)

        # action_scheme = actions.SimpleOrders()
        action_scheme = actions.BSH(cash, asset)
        reward_scheme = rewards.SimpleProfit(5)

        portfolio = Portfolio(USD, [cash, asset])

        env = default.create(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            max_allowed_loss=self.stop_loss,
            feed=data_feed,
            window_size=self.window_size)

        return env

    def observation(self, observation):
        self.full = observation
        new_obs = observation[-1]
        return new_obs


# class SamSingleIntraDay(SamSingle):
#
#     def __init__(self,
#                  data_pickle_name="1m_data",
#                  crypto_to_trade='BTC',
#                  stop_loss=0.3,
#                  window_size=1,
#                  start_qty=0,
#                  start_capital=10000
#                  ):
#         super().__init__(data_pickle_name="1m_data",crypto_to_trade, stop_loss, 88, start_qty, start_capital)
#
#
#     def create(self,
#                portfolio: 'Portfolio',
#                action_scheme: 'Union[actions.TensorTradeActionScheme, str]',
#                reward_scheme: 'Union[rewards.TensorTradeRewardScheme, str]',
#                feed: 'DataFeed',
#                window_size: int = 1,
#                min_periods: int = None,
#                **kwargs) -> TradingEnv:
#         """Creates the default `TradingEnv` of the project to be used in training
#         RL agents.
#
#         Parameters
#         ----------
#         portfolio : `Portfolio`
#             The portfolio to be used by the environment.
#         action_scheme : `actions.TensorTradeActionScheme` or str
#             The action scheme for computing actions at every step of an episode.
#         reward_scheme : `rewards.TensorTradeRewardScheme` or str
#             The reward scheme for computing rewards at every step of an episode.
#         feed : `DataFeed`
#             The feed for generating observations to be used in the look back
#             window.
#         window_size : int
#             The size of the look back window to use for the observation space.
#         min_periods : int, optional
#             The minimum number of steps to warm up the `feed`.
#         **kwargs : keyword arguments
#             Extra keyword arguments needed to build the environment.
#
#         Returns
#         -------
#         `TradingEnv`
#             The default trading environment.
#         """
#
#         action_scheme = actions.get(action_scheme) if isinstance(action_scheme, str) else action_scheme
#         reward_scheme = rewards.get(reward_scheme) if isinstance(reward_scheme, str) else reward_scheme
#
#         action_scheme.portfolio = portfolio
#
#         observer = observers.IntradayObserver(
#             portfolio=portfolio,
#             feed=feed,
#             renderer_feed=kwargs.get("renderer_feed", None),
#             window_size=window_size,
#             min_periods=min_periods,
#             randomize=True
#         )
#
#         stopper = stoppers.MaxLossStopper(
#             max_allowed_loss=kwargs.get("max_allowed_loss", 0.5)
#         )
#
#         renderer_list = kwargs.get("renderer", renderers.EmptyRenderer())
#
#         if isinstance(renderer_list, list):
#             for i, r in enumerate(renderer_list):
#                 if isinstance(r, str):
#                     renderer_list[i] = renderers.get(r)
#             renderer = AggregateRenderer(renderer_list)
#         else:
#             if isinstance(renderer_list, str):
#                 renderer = renderers.get(renderer_list)
#             else:
#                 renderer = renderer_list
#
#         env = TradingEnv(
#             action_scheme=action_scheme,
#             reward_scheme=reward_scheme,
#             observer=observer,
#             stopper=kwargs.get("stopper", stopper),
#             informer=kwargs.get("informer", informers.TensorTradeInformer()),
#             renderer=renderer
#         )
#         return env
#
#     def create_data_stream_and_feed(self):
#
#         prefix = self.crypto_to_trade + ":"
#         file = os.path.join("a2c_ppo_acktr/sam_env/data", self.pkl_name) + ".pkl"
#         if os.path.exists(file):
#             with open(file, "rb") as pkl:
#                 btc_historical_data = pickle.load(pkl)
#         else:
#             btc_historical_data = self.get_data().add_prefix(prefix)
#             N = mp.cpu_count()
#             chunk_sz = len(btc_historical_data) // N
#             remainder = len(btc_historical_data) % N
#             start_index = chunk_sz + remainder
#
#             chunks = [(btc_historical_data[:start_index], prefix)]
#             chunks += [(btc_historical_data[i-100: i+chunk_sz], prefix) for i in range(start_index, len(btc_historical_data), chunk_sz)]
#
#             with mp.Pool() as pool:
#                 result = pool.starmap(self.run, chunks)
#
#             btc_historical_data = pd.concat(list(result))
#             btc_historical_data["timestamp"] = btc_historical_data[prefix + "timestamp"]
#             btc_historical_data.sort_values(by= "timestamp", ascending=True, inplace=True)
#             btc_historical_data.drop([prefix + "timestamp"], axis=1, inplace=True)
#             btc_historical_data.bfill(inplace=True)
#             btc_historical_data.replace([np.inf, -np.inf], 0, inplace=True)
#
#             with open(file, "wb") as pkl:
#                 pickle.dump(btc_historical_data, pkl)
#
#         btc_historical_data["timestamp"] = pd.to_datetime(btc_historical_data['timestamp'], unit='s')
#         bitfinex_btc_streams = [Stream.source(btc_historical_data["timestamp"], dtype="float").rename("timestamp")]
#
#         # btc_historical_data = btc_historical_data.iloc[:, :12]
#         with NameSpace("Bitfinex"):
#             bitfinex_btc_streams += [
#                 Stream.source(list(btc_historical_data[c]), dtype="float").rename(c) for c in btc_historical_data.columns
#                 if c!= "timestamp"]
#
#         data_feed = DataFeed(bitfinex_btc_streams)
#         data_feed.compile()
#
#         return data_feed, btc_historical_data
#
#     def create_single_tick_env(self):
#         data_feed, btc_historical_data = self.create_data_stream_and_feed()
#         close_prices = btc_historical_data[self.crypto_to_trade + ":close"]
#
#         # simulated exchange
#         bitfinex = Exchange("bitfinex", service=execute_order)(
#             Stream.source(list(close_prices), dtype="float").rename("USD-" + self.crypto_to_trade))
#
#         cash = Wallet(bitfinex, self.start_capital * USD)
#         asset = Wallet(bitfinex, self.start_qty * BTC)
#
#         action_scheme = actions.SimpleOrders()
#         reward_scheme = rewards.SimpleProfit(20)
#
#         portfolio = Portfolio(USD, [cash, asset])
#
#         env = self.create(
#             portfolio=portfolio,
#             action_scheme=action_scheme,
#             reward_scheme=reward_scheme,
#             max_allowed_loss=self.stop_loss,
#             feed=data_feed,
#             window_size=self.window_size
#         )
#
#         return env


class SamSingle2D(SamSingle):

    def __init__(self, crypto_to_trade='BTC', stop_loss=0.3, start_qty=0, start_capital=10000):
        super().__init__("", crypto_to_trade, stop_loss, 88, start_qty, start_capital)

        self.observation_space.shape = [1, 88, self.observation_space.shape]

    def observation(self, observation):
        return self.full
