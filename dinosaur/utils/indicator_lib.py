import numpy as np
from scipy import stats


class BasicIndicators(object):
    """
    收益序列的基础指标的计算，包括
    1. 年化收益率
    2. 年化波动率
    3. 最高单频收益
    4. 最低单频收益
    5. 累计净值序列
    6. 最终累计收益
    7. 累计最大累计收益
    8. 最大回撤
    9. 下行风险
    10. 上行风险
    11. 收益序列峰度
    12. 收益序列偏度
    13. 收益序列胜率
    """

    def __init__(self, data, freq_multi=252):
        """
         模块初始化
        :param data: 1-D or 2-D alike data type that can be transformed to numpy ndarray
            收益序列，可以是一维也可以是二维的数据，计算时会转化为numpy ndarray
        :param freq_multi: int, default 252
            年化乘数，即日度频率数据为252，周度频率为52，月度频率为12，周度频率为4
        """
        self._data = data
        self._multiplier = freq_multi

    @property
    def annualized_return(self):
        """
        年化收益率
        :return: numpy ndarray
        """
        return np.power(np.prod(self._data + 1, axis=0), self._multiplier / self._data.shape[0]) - 1

    @property
    def annualized_vol(self):
        """
        年化波动率
        :return: numpy ndarray
        """
        return np.std(self._data, axis=0) * np.sqrt(self._multiplier)

    @property
    def best(self):
        """
        最高单频收益
        :return: numpy ndarray
        """
        return np.max(self._data, axis=0)

    @property
    def worst(self):
        """
        最低单频收益
        :return: numpy ndarray
        """
        return np.min(self._data, axis=0)

    @property
    def cumulative_returns(self):
        """
        累计净值序列
        :return: numpy ndarray
        """
        return np.cumprod(self._data + 1, axis=0)

    @property
    def final_cumulative_return(self):
        """
        最终累计收益
        :return: numpy ndarray
        """
        return self.cumulative_returns[-1] - 1

    @property
    def cumulative_max(self):
        """
        累计最大累计收益
        :return: numpy ndarray
        """
        return np.maximum.accumulate(self.cumulative_returns, axis=0)

    @property
    def max_draw_down(self):
        """
        最大回撤
        :return:  numpy ndarray
        """
        return np.min(np.divide(self.cumulative_returns - self.cumulative_max, self.cumulative_max), axis=0)

    @property
    def down_side_risk(self):
        """
        下行风险
        :return:  numpy ndarray
        """
        down_side_data = self._data.copy()
        down_side_data[down_side_data > 0] = 0
        return np.std(down_side_data, axis=0) * np.sqrt(self._multiplier)

    @property
    def up_side_risk(self):
        """
        上行风险
        :return:  numpy ndarray
        """
        down_side_data = self._data.copy()
        down_side_data[down_side_data < 0] = 0
        return np.std(down_side_data, axis=0) * np.sqrt(self._multiplier)

    @property
    def skewness(self):
        """
        收益序列峰度
        :return:  numpy ndarray
        """
        return np.array(stats.skew(self._data, axis=0))

    @property
    def kurtosis(self):
        """
        收益序列偏度
        :return: numpy ndarray
        """
        return np.array(stats.kurtosis(self._data, axis=0))

    @property
    def winning_ratio(self):
        """
        收益序列胜率
        :return: numpy ndarray
        """
        return np.nansum(self._data >= 0, axis=0)/self._data.shape[0]


class TailRisk(object):
    """
    收益序列分布的尾部风险统计指标计算，包括
    1. 在险价值（VaR）
    2. 条件在险价值（CVaR）
    3. 尾部风险——Bing Liang (2006)
    """
    def __init__(self, data, alpha=0.05):
        """
        模块初始化
        :param data: 1-D or 2-D alike data type that can be transformed to numpy ndarray
            收益序列，可以是一维也可以是二维的数据，计算时会转化为numpy ndarray
        :param alpha: float, default 0.05
            置信区间系数，即置信区间概率为1-alpha
        """
        self._alpha = alpha
        self._data = data
        pass

    @property
    def value_at_risk(self):
        """
        在险价值（VaR）
        :return: numpy ndarray
        """
        sorted_data = np.sort(self._data.copy(), axis=0)
        quantile = int(np.ceil(self._data.shape[0] * self._alpha))
        return sorted_data[quantile]

    @property
    def conditional_value_at_risk(self):
        """
        在险价值（VaR）
        :return: numpy ndarray
        """
        positions = np.int(self._data <= self.value_at_risk)
        cross_sum = np.sum(np.multiply(self._data, positions), axis=0)
        dividers = np.sum(positions, axis=0)
        return np.divide(cross_sum, dividers)

    @property
    def tail_risk(self):
        """
        尾部风险——Reference Bing Liang (2006)
        :return: numpy ndarray
        """
        positions = np.int(self._data <= self.value_at_risk)
        positional_diff = np.power(np.multiply(self._data - np.mean(self._data, axis=0), positions), 2)
        position_sum = np.sum(positions, axis=0)
        return np.sqrt(np.divide(np.sum(positional_diff, axis=0), position_sum))


class PassiveIndicators(object):
    """
    通过无风险利率来衡量组合表现的指标，包括
    1. 相对无风险收益的超额收益
    2. 夏普比率
    3. 卡玛比率
    4. 索提诺比率
    """

    def __init__(self, data, risk_free=0.015, freq_multi=252):
        """
        模块初始化
        :param data: 1-D or 2-D alike data type that can be transformed to numpy ndarray
            收益序列，可以是一维也可以是二维的数据，计算时会转化为numpy ndarray
        :param risk_free: float, default 0.015
            无风险收益率，默认无风险收益率为1.5%
        :param freq_multi:  int, default 252
            年化乘数，即日度频率数据为252，周度频率为52，月度频率为12，周度频率为4
        """
        self._basic_indicators = BasicIndicators(data, freq_multi)
        self._risk_free = risk_free

    @property
    def access_return(self):
        """
        相对无风险收益的超额收益
        :return: numpy ndarray
        """
        return self._basic_indicators.annualized_return - self._risk_free

    @property
    def sharpe_ratio(self):
        """
        夏普比率
        :return: numpy ndarray
        """
        return np.divide(self.access_return, self._basic_indicators.annualized_vol)

    @property
    def calmar_ratio(self):
        """
        卡玛比率
        :return: numpy ndarray
        """
        max_dd = abs(self._basic_indicators.max_draw_down)
        return np.divide(self.access_return, max_dd)  # .replace(np.inf, np.nan)

    @property
    def sortino_ratio(self):
        """
        索提诺比率
        :return: numpy ndarray
        """
        down_side_risk = self._basic_indicators.down_side_risk
        # down_side_risk[down_side_risk == 0] = np.inf
        return np.divide(self.access_return, down_side_risk)


class ActiveIndicators(object):
    """
    通过对比基准（benchmark）衡量组合表现的指标，包括
    1. 超额收益年化收益
    2. 超额收益年化波动率
    3. 年化信息比率

    """

    def __init__(self, data, benchmark, freq_multi):
        self._data = data
        self._bench_mark = benchmark
        self._multiplier = freq_multi
        pass

    @property
    def active_annual_return(self):
        """
        超额收益年化收益
        :return: numpy ndarrays
        """
        return BasicIndicators(self._data - self._bench_mark, self._multiplier).annualized_return

    @property
    def active_annual_vol(self):
        """
        超额收益年化波动率
        :return: numpy ndarrays
        """
        return BasicIndicators(self._data - self._bench_mark, self._multiplier).annualized_vol

    @property
    def annual_information_ratio(self):
        """
        年化信息比率
        :return: numpy ndarrays
        """
        return np.divide(self.active_annual_return, self.active_annual_vol)
