import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from IPython import display
import warnings
warnings.filterwarnings("ignore")

# 导入路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINRL_META_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'FinRL-Meta'))
print(f'FINRL_META_DIR: {FINRL_META_DIR}, SCRIPT_DIR: {SCRIPT_DIR}')
sys.path.insert(0, FINRL_META_DIR)

# 导入必要的模块
from meta import config
from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource
from meta.data_processors.akshare import Akshare
from meta.env_stock_trading.env_stocktrading_China_A_shares import StockTradingEnv
from agents.stablebaselines3_models import DRLAgent
from stable_baselines3 import DDPG, A2C, PPO, SAC


class InvestmentAdvisor:
    """
    使用训练好的强化学习模型生成投资建议
    """
    def __init__(self, model_path, ticker_list=None, token=None):
        """
        初始化投资顾问

        参数:
            model_path: 训练好的模型路径
            ticker_list: 股票列表
            token: AKShare API令牌
        """
        self.model_path = model_path

        # 如果没有提供股票列表，使用默认列表
        self.ticker_list = ticker_list or [
            "600000.SH", "600009.SH", "600016.SH", "600028.SH", "600030.SH",
            "600031.SH", "600036.SH", "600050.SH", "600104.SH", "600196.SH",
            "600276.SH", "600309.SH", "600519.SH", "600547.SH", "600570.SH"
        ]

        # 设置AKShare令牌
        self.token = token or "27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5"

        # 加载模型
        self.model = None
        self.model_type = None
        if "ddpg" in model_path.lower():
            self.model = DDPG.load(model_path)
            self.model_type = "DDPG"
        elif "a2c" in model_path.lower():
            self.model = A2C.load(model_path)
            self.model_type = "A2C"
        elif "ppo" in model_path.lower():
            self.model = PPO.load(model_path)
            self.model_type = "PPO"
        elif "sac" in model_path.lower():
            self.model = SAC.load(model_path)
            self.model_type = "SAC"
        else:
            raise ValueError("不支持的模型类型。请使用路径中包含'ddpg'、'a2c'、'ppo'或'sac'的模型文件。")

        print(f"成功加载{self.model_type}模型从: {model_path}")

    def get_latest_data(self, lookback_days=60):
        """
        获取最新的市场数据并添加技术指标

        参数:
            lookback_days: 回溯天数，用于计算技术指标

        返回:
            处理后的数据框
        """
        # 计算日期范围
        end_date = dt.datetime.now().strftime("%Y-%m-%d")
        start_date = (dt.datetime.now() - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        print(f"获取从 {start_date} 到 {end_date} 的市场数据...")

        # 获取和处理数据
        kwargs = {"token": self.token}
        processor = DataProcessor(
            data_source=DataSource.akshare,
            start_date=start_date,
            end_date=end_date,
            time_interval="daily",
            **kwargs
        )

        # 下载和清洗数据
        processor.download_data(ticker_list=self.ticker_list)
        processor.clean_data()
        processor.fillna()

        # 添加与训练时相同的技术指标
        processor.add_technical_indicator(config.INDICATORS)
        processor.fillna()

        return processor.dataframe

    def generate_investment_advice(self, initial_amount=1000000, cash_reserve_pct=0.1):
        """
        生成投资建议

        参数:
            initial_amount: 初始投资金额
            cash_reserve_pct: 现金储备百分比

        返回:
            投资建议和可视化图表
        """
        # 获取最新数据
        latest_data = self.get_latest_data()
        if latest_data.empty:
            print("无法获取市场数据。请检查您的网络连接和AKShare令牌。")
            return None

        print(f"成功获取了 {len(latest_data)} 条市场数据记录")

        # 准备环境
        stock_dimension = len(latest_data.tic.unique())
        state_space = stock_dimension * (len(config.INDICATORS) + 2) + 1

        env_kwargs = {
            "stock_dim": stock_dimension,
            "hmax": 1000,
            "initial_amount": initial_amount,
            "buy_cost_pct": 0.00025 + 0.00004,
            "sell_cost_pct": 0.00025 + 0.00100 + 0.00002 + 0.00004,
            "reward_scaling": 1e-4,
            "state_space": state_space,
            "action_space": stock_dimension,
            "tech_indicator_list": config.INDICATORS,
            "print_verbosity": 0,  # 减少输出
            "initial_buy": False,
            "hundred_each_trade": True,
        }

        # 创建预测环境
        e_trade_gym = StockTradingEnv(df=latest_data, **env_kwargs)

        # 使用模型进行预测
        print(f"使用{self.model_type}模型生成投资建议...")
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=self.model,
            environment=e_trade_gym
        )

        # 获取最新一天的动作建议
        if df_actions.empty:
            print("模型没有生成任何交易动作。")
            return None

        latest_actions = df_actions.iloc[-1].to_dict()

        # 移除date列
        if "date" in latest_actions:
            del latest_actions["date"]

        # 计算投资金额
        investable_amount = initial_amount * (1 - cash_reserve_pct)
        cash_reserve = initial_amount * cash_reserve_pct

        # 生成投资建议
        advice = []
        total_weight = sum(latest_actions.values())

        if total_weight <= 0:
            print("警告: 模型建议全部清仓或市场风险过高。建议增加现金持有比例。")
            return None

        for tic, weight in sorted(latest_actions.items(), key=lambda x: x[1], reverse=True):
            if weight > 0:
                # 计算每只股票的投资金额
                stock_amount = (weight / total_weight) * investable_amount
                advice.append({
                    "股票代码": tic,
                    "配置权重": weight / total_weight,
                    "建议投资金额": stock_amount,
                    "预计可买入股数": int(stock_amount / latest_data[latest_data.tic == tic].close.iloc[-1] / 100) * 100
                })

        # 创建投资建议数据框
        advice_df = pd.DataFrame(advice)

        # 添加指定现金持有
        cash_advice = pd.DataFrame([{
            "股票代码": "现金",
            "配置权重": cash_reserve_pct,
            "建议投资金额": cash_reserve,
            "预计可买入股数": None
        }])

        final_advice = pd.concat([advice_df, cash_advice], ignore_index=True)

        # 生成可视化图表
        self._plot_allocation(final_advice)

        return final_advice

    def _plot_allocation(self, advice_df):
        """生成资产配置饼图"""
        plt.figure(figsize=(12, 8))

        # 饼图显示权重配置
        plt.subplot(1, 2, 1)
        labels = advice_df["股票代码"].tolist()
        sizes = advice_df["配置权重"].tolist()
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'{self.model_type}模型投资组合权重配置')

        # 条形图显示投资金额
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(labels))
        amounts = advice_df["建议投资金额"].tolist()
        plt.barh(y_pos, amounts)
        plt.yticks(y_pos, labels)
        plt.xlabel('投资金额 (元)')
        plt.title('各资产投资金额')

        plt.tight_layout()
        plt.savefig("投资建议.png")
        plt.show()

    def rebalance_suggestions(self, current_holdings, initial_amount=1000000, cash_reserve_pct=0.1):
        """
        根据当前持仓提供再平衡建议

        参数:
            current_holdings: 字典，键为股票代码，值为持有数量
            initial_amount: 当前投资组合总价值
            cash_reserve_pct: 建议的现金储备比例

        返回:
            再平衡建议
        """
        # 获取最新数据和价格
        latest_data = self.get_latest_data(lookback_days=30)  # 较短的回溯期足够获取当前价格

        # 获取最新价格
        latest_prices = {}
        for tic in current_holdings.keys():
            if tic != "现金" and tic in latest_data.tic.unique():
                latest_prices[tic] = latest_data[latest_data.tic == tic].close.iloc[-1]

        # 计算当前持仓价值
        current_values = {}
        total_value = 0
        for tic, shares in current_holdings.items():
            if tic == "现金":
                current_values[tic] = shares  # 现金值直接使用
                total_value += shares
            elif tic in latest_prices:
                current_values[tic] = shares * latest_prices[tic]
                total_value += current_values[tic]

        # 获取新的建议
        if total_value == 0:
            print("当前持仓总价值为0，无法提供再平衡建议。")
            return None

        new_advice = self.generate_investment_advice(initial_amount=total_value, cash_reserve_pct=cash_reserve_pct)

        if new_advice is None:
            return None

        # 创建再平衡建议
        rebalance = []
        for _, row in new_advice.iterrows():
            tic = row["股票代码"]
            suggested_value = row["建议投资金额"]

            current_value = current_values.get(tic, 0)
            value_diff = suggested_value - current_value

            action = "持有"
            shares_change = 0

            if tic != "现金":
                if value_diff > 0:
                    action = "买入"
                    shares_change = int(value_diff / latest_prices.get(tic, 1) / 100) * 100
                elif value_diff < 0:
                    action = "卖出"
                    shares_change = -int(abs(value_diff) / latest_prices.get(tic, 1) / 100) * 100
            else:
                # 现金调整
                shares_change = value_diff
                if value_diff > 0:
                    action = "增加现金"
                elif value_diff < 0:
                    action = "减少现金"

            rebalance.append({
                "股票代码": tic,
                "当前持仓价值": current_value,
                "建议配置价值": suggested_value,
                "价值差额": value_diff,
                "操作": action,
                "操作数量": shares_change if tic != "现金" else abs(shares_change)
            })

        return pd.DataFrame(rebalance)


# 使用示例
if __name__ == "__main__":
    # 加载训练好的模型
    model_path = os.path.join(SCRIPT_DIR, "FinRL-Meta/trained_models/ddpg/best_model.zip")

    # 如果模型文件不存在，提醒用户
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请确保您已经训练好了模型，或者调整路径指向正确的模型文件。")
        model_path = input("请输入训练好的模型路径: ")

    # 创建投资顾问
    advisor = InvestmentAdvisor(model_path=model_path)

    # 生成投资建议
    print("\n生成新的投资组合建议...")
    advice = advisor.generate_investment_advice(initial_amount=1000000, cash_reserve_pct=0.2)

    if advice is not None:
        print("\n投资建议:")
        print(advice)
        print("\n投资建议已保存到 '投资建议.png'")

    # 示例: 再平衡现有投资组合
    print("\n\n为现有投资组合提供再平衡建议...")
    # 示例当前持仓
    current_holdings = {
        "600000.SH": 1000,  # 持有1000股浦发银行
        "600036.SH": 500,   # 持有500股招商银行
        "600519.SH": 100,   # 持有100股贵州茅台
        "现金": 200000      # 持有20万现金
    }

    rebalance = advisor.rebalance_suggestions(
        current_holdings=current_holdings,
        initial_amount=None,  # 会自动计算当前持仓总价值
        cash_reserve_pct=0.2
    )

    if rebalance is not None:
        print("\n再平衡建议:")
        print(rebalance)
