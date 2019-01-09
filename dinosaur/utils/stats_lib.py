from collections import OrderedDict
from pandas import Series, to_datetime, DataFrame, DateOffset
from pandas.tseries.offsets import YearBegin
from numpy import nan

from dinosaur.utils.indicator_lib import BasicIndicators, PassiveIndicators


class SummaryStatistic(object):
    """
    对组合或收益序列历史表现的统计指标的计算，返回以下指标
    1. CumReturn_ITD %: 成立以来累计收益率%
    2. CumReturn_YTD %: 今年以来累计收益率%
    3. CumReturn_3M %: 近3个月来累计收益%
    4. CumReturn_6M %: 近6个月来累计收益率%
    5. CumReturn_1Y %: 近1年来累计的收益率%
    6. AnnReturn %: 年化收益率%
    7. AnnVol %: 年化波动率%
    8. MaxDrawDown %: 最大回撤%
    9. MonthlyWiningRatio %: 自然月度胜率%
    10. QuarterlyWiningRatio %: 自然季度胜率%
    11. SharpeRatio: 夏普比率
    12. CalmarRatio: 卡玛比率
    13. SortinoRatio: 索提诺比率
    14. StartDate: 开始时间
    15. EndDate: 结束时间

    """

    def __init__(self, data_df, risk_free=0.015, is_nav=True):
        """
        模块初始化
        :param data_df: pandas DataFrame
            组合收益序列或净值序列
        :param risk_free: float, default 0.015
            无风险收益率（年化），默认为1.5%
        :param is_nav: bool, default True
            是否为净值序列，默认是
        """
        self._risk_free = risk_free
        self._items_order = []
        self._holder = {col: self._fetch_slice(data_df.loc[:, col], is_nav=is_nav) for col in data_df.columns}

    def _fetch_slice(self, ser, is_nav=True, nan_filled=True):
        """
        对单个序列进行计算
        :param ser: pandas Series
            单个组合收益或净值序列
        :param is_nav: bool, default True
            是否为净值，默认为否
        :param nan_filled: bool, default True
            是否对于净值缺失进行填补，默认是
        :return: collections OrderedDict
            统计指标的dict
        """
        assert isinstance(ser, Series)
        flesh = ser.dropna().sort_index()
        if nan_filled:
            valid = ser.loc[flesh.index[0]:flesh.index[-1]].fillna(method='ffill')
        else:
            valid = flesh

        valid = valid.pct_change().fillna(0) if is_nav else valid
        valid.index = to_datetime(valid.index)
        start = min(valid.index)
        end = max(valid.index)
        multi = 365/(valid.__len__() - 1)*(end - start).days

        holder = OrderedDict()
        holder.update({'CumReturn_ITD %': round(100*BasicIndicators(valid, multi).final_cumulative_return, 2)})
        holder.update({
            'CumReturn_YTD %': round(100*BasicIndicators(valid.loc[end - YearBegin():end], multi).final_cumulative_return, 2)
        })
        if end - DateOffset(month=3) >= start:
            holder.update({'CumReturn_3M %': round(100*BasicIndicators(valid.loc[end - DateOffset(months=3):end], multi).
                          final_cumulative_return, 2)})
        else:
            holder.update({'CumReturn_3M %': nan})

        if end - DateOffset(month=6) >= start:
            holder.update({'CumReturn_6M %': round(100*BasicIndicators(valid.loc[end - DateOffset(months=6):end], multi).
                          final_cumulative_return, 2)})
        else:
            holder.update({'CumReturn_6M %': nan})

        if end - DateOffset(month=12) >= start:
            holder.update({'CumReturn_1Y %': round(100*BasicIndicators(valid.loc[end - DateOffset(months=12):end], multi).
                          final_cumulative_return, 2)})
        else:
            holder.update({'CumReturn_1Y %': nan})

        holder.update({
            'AnnReturn %': round(100*BasicIndicators(valid, multi).annualized_return, 2)
        })
        holder.update({
            'AnnVol %': round(100*BasicIndicators(valid, multi).annualized_vol, 2)
        })
        holder.update({
            'MaxDrawDown %': round(-100*BasicIndicators(valid, multi).max_draw_down, 2)
        })
        holder.update({
            'MonthlyWiningRatio %': round(100*BasicIndicators((valid + 1).resample('1M').prod() - 1, 12).winning_ratio, 2)
        })
        holder.update({
            'QuarterlyWiningRatio %': round(100*BasicIndicators((valid + 1).resample('1Q').prod() - 1, 4).winning_ratio, 2)
        })
        holder.update({
            'SharpeRatio': round(PassiveIndicators(valid, self._risk_free, multi).sharpe_ratio, 2)
        })
        holder.update({
            'CalmarRatio': round(PassiveIndicators(valid, self._risk_free, multi).calmar_ratio, 2)
        })
        holder.update({
            'SortinoRatio': round(PassiveIndicators(valid, self._risk_free, multi).sortino_ratio, 2)
        })

        holder.update({'StartDate': start.strftime('%Y-%m-%d')})
        holder.update({'EndDate': end.strftime('%Y-%m-%d')})
        if self._items_order.__len__() == 0:
            self._items_order.extend(holder.keys())
        return holder

    @property
    def summary(self):
        """
        输出统结果
        :return: pandas DataFrame
        结构如下
                                   group_1     ...      equal_weight
        CumReturn_ITD %              -12.5     ...             23.23
        CumReturn_YTD %              -8.76     ...              -7.8
        CumReturn_3M %               -6.02     ...             -3.87
        CumReturn_6M %               -8.18     ...             -7.25
        CumReturn_1Y %               -6.34     ...             -1.04
        AnnReturn %                  -3.45     ...              5.64
        AnnVol %                     32.11     ...             32.39
        MaxDrawDown %                42.64     ...             41.56
        MonthlyWiningRatio %         48.54     ...             55.34
        QuarterlyWiningRatio %       48.57     ...             54.29
        SharpeRatio                  -0.15     ...              0.13
        CalmarRatio                  -0.12     ...               0.1
        SortinoRatio                 -0.23     ...              0.19
        StartDate               2009-12-01     ...        2009-12-01
        EndDate                 2018-06-29     ...        2018-06-29
        """
        return DataFrame(self._holder).reindex(index=self._items_order)


if __name__ == '__main__':
    import numpy as np
    from pandas import date_range
    test_data = DataFrame(np.random.randn(400, 20)/100, date_range(start='2008-01-01', periods=400, freq='D'))
    print(SummaryStatistic(test_data).summary)

