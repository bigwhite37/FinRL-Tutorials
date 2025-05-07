import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display

display.set_matplotlib_formats("svg")

# Add FinRL-Meta directory to the import path for meta imports
import sys
import os
# Ensure FinRL-Meta/meta is on the import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINRL_META_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'FinRL-Meta'))
print(f'FINRL_META_DIR: {FINRL_META_DIR}, SCRIPT_DIR: {SCRIPT_DIR}')
sys.path.insert(0, FINRL_META_DIR)

from meta import config
from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource
from main import check_and_make_directories
from meta.data_processors.akshare import Akshare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import (
    StockTradingEnv,
)
from agents.stablebaselines3_models import DRLAgent
import os
from typing import List
from argparse import ArgumentParser
from meta import config
from meta.config_tickers import DOW_30_TICKER
from meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)
import pyfolio
from pyfolio import timeseries

pd.options.display.max_columns = None

print("ALL Modules have been imported!")


### Create folders

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)


### Download data, cleaning and feature engineering

# 根据市值和流动性筛选股票
import akshare as ak

# 获取A股所有股票列表
stock_info_a_code_name = ak.stock_info_a_code_name()
stock_info_a_code_name = stock_info_a_code_name[:1000]

# 获取股票流动性和市值数据
def get_stocks_by_criteria(min_market_cap=500, min_daily_volume=50000000):
    selected_stocks = []
    for idx, row in stock_info_a_code_name.iterrows():
        code = row["code"]
        # 获取股票基本信息
        stock_info = ak.stock_individual_info_em(symbol=code)
        print(f"处理股票: {code} - {row['name']}")

        # 获取市值（总市值单位为元，转换为亿元）
        if '总市值' in stock_info['item'].values:
            market_cap_raw = stock_info.loc[stock_info['item'] == '总市值', 'value'].values[0]
            market_cap = float(market_cap_raw) / 100000000  # 元转亿元
        else:
            print(f"警告: 股票 {code} 没有找到总市值数据")
            continue

        # 获取成交量数据（通过历史数据计算平均值）
        from datetime import datetime, timedelta
        end_date = datetime.today().strftime("%Y%m%d")
        start_date = (datetime.today() - timedelta(days=30)).strftime("%Y%m%d")
        # 获取最近30个交易日数据
        stock_hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="")
        if not stock_hist.empty and '成交量' in stock_hist.columns:
            daily_volume = stock_hist['成交量'].mean()
        else:
            print(f"警告: 股票 {code} 没有找到成交量数据")
            continue

        # 打印调试信息
        print(f"  市值: {market_cap:.2f}亿元, 日均成交量: {daily_volume:.0f}")

        # 筛选条件
        if market_cap >= min_market_cap and daily_volume >= min_daily_volume:
            # 转换为适合akshare的格式
            if code.startswith('6'):
                formatted_code = f"{code}.SH"
            else:
                formatted_code = f"{code}.SZ"
            selected_stocks.append(formatted_code)
            print(f"  已选择: {formatted_code}")

        # 限制数量
        if len(selected_stocks) >= 30:
            print(f"已达到30只股票限制，停止筛选")
            break

    print(f"共选择了 {len(selected_stocks)} 只股票")
    return selected_stocks

# 使用函数获取股票
ticker_list = get_stocks_by_criteria()
print(f"ticker_list: {ticker_list}")

TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2019-08-01"
TRADE_START_DATE = "2019-08-01"
TRADE_END_DATE = "2020-01-03"

TIME_INTERVAL = "daily"
kwargs = {}
kwargs["token"] = "27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5"
p = DataProcessor(
    data_source=DataSource.akshare,
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL,
    **kwargs,
)


# download and clean
p.download_data(ticker_list=ticker_list)
p.clean_data()
p.fillna()

# add_technical_indicator
p.add_technical_indicator(config.INDICATORS)
p.fillna()
print(f"p.dataframe: {p.dataframe}")


### Split traning dataset

train = p.data_split(p.dataframe, TRAIN_START_DATE, TRAIN_END_DATE)
print(f"len(train.tic.unique()): {len(train.tic.unique())}")

print(f"train.tic.unique(): {train.tic.unique()}")

print(f"train.head(): {train.head()}")

print(f"train.shape: {train.shape}")

stock_dimension = len(train.tic.unique())
state_space = stock_dimension * (len(config.INDICATORS) + 2) + 1
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

### Train

env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 1000000,
    # 买入时只收 佣金 + 结算监管费
    "buy_cost_pct": 0.00025 + 0.00004,   # ≈0.00029 (0.029%)
    # 卖出时加上印花税和（沪市）过户费
    "sell_cost_pct": 0.00025 + 0.00100 + 0.00002 + 0.00004,  # ≈0.00131 (0.131%)
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": True,
    "hundred_each_trade": True,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)

## DDPG

env_train, _ = e_train_gym.get_sb_env()
print(f"print(type(env_train)): {print(type(env_train))}")

agent = DRLAgent(env=env_train)
DDPG_PARAMS = {
    "batch_size": 256,
    "buffer_size": 50000,
    "learning_rate": 0.0005,
    "action_noise": "normal",
}
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
model_ddpg = agent.get_model(
    "ddpg", model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS
)

trained_ddpg = agent.train_model(
    model=model_ddpg, tb_log_name="ddpg", total_timesteps=100000
)

# 创建保存目录（如果不存在）
model_save_dir = os.path.join(FINRL_META_DIR, "trained_models", "ddpg")
os.makedirs(model_save_dir, exist_ok=True)

# 保存模型
model_save_path = os.path.join(model_save_dir, "best_model.zip")
trained_ddpg.save(model_save_path)
print(f"模型已保存到: {model_save_path}")

## A2C

agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")

trained_a2c = agent.train_model(
    model=model_a2c, tb_log_name="a2c", total_timesteps=50000
)

### Trade

trade = p.data_split(p.dataframe, TRADE_START_DATE, TRADE_END_DATE)
env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.00025 + 0.00004,   # ≈0.00029 (0.029%)
    "sell_cost_pct": 0.00025 + 0.00100 + 0.00002 + 0.00004,  # ≈0.00131 (0.131%)
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": False,
    "hundred_each_trade": True,
}
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg, environment=e_trade_gym
)

df_actions.to_csv("action.csv", index=False)
print(f"df_actions: {df_actions}")

### Backtest

# matplotlib inline
plotter = ReturnPlotter(df_account_value, trade, TRADE_START_DATE, TRADE_END_DATE)
# plotter.plot_all()

plotter.plot()

# matplotlib inline
# # ticket: SSE 50：000016
# plotter.plot("000016")

#### Use pyfolio

# CSI 300
baseline_df = plotter.get_baseline("399300")


daily_return = plotter.get_return(df_account_value)
daily_return_base = plotter.get_return(baseline_df, 'close')

perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=daily_return,
    factor_returns=daily_return_base,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
print("==============DRL Strategy Stats===========")
print(f"perf_stats_all: {perf_stats_all}")


daily_return = plotter.get_return(df_account_value)
daily_return_base = plotter.get_return(baseline_df, 'close')

perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=daily_return_base,
    factor_returns=daily_return_base,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
print("==============Baseline Strategy Stats===========")
print(f"perf_stats_all: {perf_stats_all}")
