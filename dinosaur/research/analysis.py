import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ttest_1samp
from os.path import join, exists, pardir
import matplotlib.pyplot as plt
from multiprocessing import Pool

from dinosaur.utils import ProgressBar
from dinosaur.utils.stats_lib import SummaryStatistic

__all__ = ['IndustryCycling']


class IndustryCycling(object):
    """
    分析基金行业轮动能力模块
    主要包括以下功能：
        1）读取的本地数据——基金的净值数据、行业指数数据及市场的行业市值权重
        2）依据二次规划构造的系数约束线性回归模型，各系数在(0, 1)之间，约束系数之和在(0, 1)之间，
        3）对估算仓位的中性化处理
        4）构造基金行业轮动能力指标
        5）根据信号构造的简单回测模块
    """

    def __init__(self, data_path):
        """
        模块初始化
        :param data_path: str
            数据文件夹所在路径
        """
        self._data_path = data_path
        (self._fund_return, self._industrial_indices, self._mkt_ind_weight,
         self._fund_fwd, self._ind_fwd) = self._data_init()
        print('Initialization completed.')

    def _data_init(self):
        """
        获取本地数据
        :return: tuple of pandas DataFrames
            按以下顺序返回
            1. 基金收益序列
            2. 行业指数收益序列
            3. 市场行业权重
            4. 基金按不同窗口的下期收益率
            5. 行业指数按不同窗口的下去收益率
        """
        print('Getting data...')
        fund_nav = pd.read_csv(join(self._data_path, 'fund_nav.csv'), parse_dates=['tdate']).set_index('tdate')
        fund_return = fund_nav.pct_change(fill_method=None)
        fund_return.columns.name = 'fund_code'  # 需要对fund_return的columns命名为‘fund_code’，后续需要用到
        industrial_indices = pd.read_csv(join(self._data_path, 'industrial_index.csv'),
                                         parse_dates=['tdate']).set_index('tdate')
        industrial_indices.columns = [int(x) for x in industrial_indices.columns]
        industrial_indices.columns.name = 'sw_code'  # 需要对industrial_indices的columns命名为‘sw_code’，后续需要用到
        mkt_ind_weight = pd.read_csv(join(self._data_path, 'ind_weight.csv'), parse_dates=['tdate']).set_index('tdate')
        mkt_ind_weight.columns = [int(x) for x in mkt_ind_weight.columns]

        # 如果本地没有预计算好的fund_fwd_return.csv则需要重新计算，该过程耗时较长
        if exists(join(self._data_path, 'fund_fwd_return.csv')):
            fund_fwd_return = pd.read_csv(join(self._data_path, 'fund_fwd_return.csv'),
                                          index_col=0, parse_dates=['tdate'])
        else:
            fund_fwd_return = self.fwd_return(range(1, 13), fund_return.index, fund_return)
            fund_fwd_return.to_csv(join(self._data_path, 'fund_fwd_return.csv'))

        if exists(join(self._data_path, 'ind_fwd_return.csv')):
            ind_fwd_return = pd.read_csv(join(self._data_path, 'ind_fwd_return.csv'),
                                         index_col=0, parse_dates=['tdate'])
        else:
            ind_fwd_return = self.fwd_return(range(1, 13), industrial_indices.index, industrial_indices)
            ind_fwd_return.to_csv(join(self._data_path, 'ind_fwd_return.csv'))

        return fund_return, industrial_indices, mkt_ind_weight, fund_fwd_return, ind_fwd_return

    @staticmethod
    def qp_estimate_exposure(Y, X, has_const=False):
        """
        采用二次规划的方法求解带限制的回归模型
        限制为解释变量的回归系数在 (0, 1) 之间，且系数之和在 (0, 1) 之间
        :param Y: 1-D alike data-type
            被解释变量
        :param X: 2-D alike data-type, row number should be identical with the length of Y
            解释变量
        :param has_const: bool, default False
            是否有含截距项
        :return: ndarray
            二次规划的回归系数
        """
        # 将入参Y，X均转化为numpy ndarray，其中Y必须要转化为n行1列的形式
        Y = np.array(Y).reshape(-1, 1)
        X = np.array(X)
        if has_const:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 优化的目标方程，即LSE
        def target(expo, y, x):
            return np.nansum((y - x.dot(expo.reshape(-1, 1))) ** 2)

        # 参数限制，依照研报要求，即为所有参数合在0~1之间
        def constraint(x):
            return 1 - np.nansum(x[1:])

        # 虽然在研报中未明确说明，但是将单个行业配置比率现在0~1之间为更符合实际的上下界限制
        # 以下构造scipy.optimize.minimize中'SLSQP'优化方法带约束的条件，其中优化起点为等权
        bounds = ((0, 1),) * (X.shape[1]) if not has_const else ((None, None), ) + ((0, 1),) * (X.shape[1] - 1)
        cons_dict = {'type': 'ineq', 'fun': constraint}
        first_try = np.ones((X.shape[1], 1)) / X.shape[1]
        opt = minimize(fun=target, x0=first_try, args=(Y, X,),
                       method='SLSQP', bounds=bounds,
                       constraints=cons_dict)

        if opt.success:
            if has_const:
                return opt.x[1:]
            else:
                return opt.x
        else:
            if has_const:
                return first_try[1:]
            else:
                return first_try

    def get_fund_allocation_est(self, obs_months=3, renew=False, multi=False):
        """
        获取所有基金的行业仓位估算，如果本地不存在则会重新计算并保存本地
        :param obs_months: int, default 3
            以过去多久时间（多少个月）的数据来进行估算基金行业配合仓位，默认为3个月
        :param renew: bool, False
            是否重新计算
        :param multi: bool, False
            是否使用multi processing
        :return: pandas DataFrame
            返回基金行业配置仓位历史，结构如下

                allocation_est  fund_code  sw_code      tdate
            0          0.0334  000056.OF   110000 2014-07-31
            1          0.0253  000056.OF   210000 2014-07-31
            2          0.0245  000056.OF   220000 2014-07-31
            3          0.0420  000056.OF   230000 2014-07-31
            4          0.0237  000056.OF   240000 2014-07-31

        """
        file_dir = join(self._data_path, 'fund_allocation_est_{}.csv'.format(obs_months))
        if exists(file_dir) and not renew:
            return pd.read_csv(file_dir, index_col=0, parse_dates=['tdate'])
        else:
            data = self.cal_all_funds_allocations(obs_months=obs_months, multi=multi)
            data.to_csv(file_dir)
            return data

    def single_fund_allocation(self, fund_code, obs_months=3, month_step=1, bar=None):
        """
         计算单只基金的行业仓位估算历史
        :param fund_code: string
            基金代码，需为本地基金数据中的列名
        :param obs_months: int, default 3
            以过去多久时间（多少个月）的数据来进行估算基金行业配合仓位，默认为3个月
        :param month_step: int, default 1
            滚动窗口（月），默认为1个月
        :param bar: subclass of ProgressBar, default None
            内置的进度条，如果需要在处理单只基金时显示进度条，加入此项
        :return: pandas DataFrame
            返回基金行业配置仓位历史，结构如下

                allocation_est  fund_code  sw_code      tdate
            0          0.0334  000056.OF   110000 2014-07-31
            1          0.0253  000056.OF   210000 2014-07-31
            2          0.0245  000056.OF   220000 2014-07-31
            3          0.0420  000056.OF   230000 2014-07-31
            4          0.0237  000056.OF   240000 2014-07-31
        """
        f_return = self._fund_return.loc[:, fund_code].dropna()  # 获取对应基金的历史收益
        # 将基金收益数据与行业指数收益对其时间并合并
        combined = f_return.to_frame().join(self._industrial_indices.reindex(index=f_return.index))
        date_reference = self._fund_return.index
        # 时间起点为合并后时间轴（index）的最后一个时间戳，如果该时间戳不是全部基金收益序列时间轴的该月最后
        # 一个有效日期，则表示该基金在该月月中便清算或停止，故向前移一个月末作为起始时间
        start_date = combined.index[-1]
        if start_date < max(date_reference[date_reference <= start_date + pd.offsets.MonthEnd()]):
            start_date = max(date_reference[date_reference <= start_date - pd.offsets.MonthEnd()])
        obs_date = start_date

        # 如果第一个其实时间戳向前obs_months个月份的日期早于combined的第一个时间戳，表示其业绩不足计算行业
        # 配置的要求，返回空DataFrame
        if obs_date - pd.offsets.MonthBegin(obs_months) < combined.index[0]:
            return fund_code, pd.DataFrame(columns=['tdate', 'sw_code', 'allocation_est'])
        holder = {}
        while True:
            try:
                assert isinstance(bar, ProgressBar)
                bar.show(fund_code, False)
            except AssertionError:
                pass

            inception = obs_date - pd.offsets.MonthBegin(obs_months)
            # 按窗口取数据，每个子DataFrame的第一列将作为回归的被解释变量，之后所有列即行业收益率作为解释变量，参与以下
            # 优化
            obs_df = combined.loc[inception:obs_date, :]
            # 将qp_estimate_exposure优化结果整合合并
            opt = pd.Series(self.qp_estimate_exposure(obs_df.iloc[:, 0], obs_df.iloc[:, 1:], has_const=False),
                            index=combined.columns[1:],
                            name='expo_est').round(4)
            holder.update({obs_date: opt})
            obs_date = max(date_reference[date_reference <= obs_date - pd.offsets.MonthEnd(month_step)])
            if obs_date - pd.offsets.MonthBegin(obs_months) < combined.index[0]:
                break
        ind_holding_est = pd.DataFrame(holder).T.sort_index().stack().reset_index().rename(columns={
            'level_0': 'tdate', 'level_1': 'sw_code', 0: 'allocation_est'})
        ind_holding_est['fund_code'] = fund_code

        try:
            assert isinstance(bar, ProgressBar)
            bar.show(fund_code, True)
        except AssertionError:
            pass

        return fund_code, ind_holding_est

    def cal_all_funds_allocations(self, obs_months=3, multi=True):
        """
        计算全部基金的行业仓位估算
        :param obs_months: int, default None
            以过去多久时间（多少个月）的数据来进行估算基金行业配合仓位，默认为3个月
        :param multi: bool, default True
            是否使用multi processing
        :return:
            返回基金行业配置仓位历史，结构如下

                allocation_est  fund_code  sw_code      tdate
            0          0.0334  000056.OF   110000 2014-07-31
            1          0.0253  000056.OF   210000 2014-07-31
            2          0.0245  000056.OF   220000 2014-07-31
            3          0.0420  000056.OF   230000 2014-07-31
            4          0.0237  000056.OF   240000 2014-07-31
        """
        fund_list = list(self._fund_return.columns)
        bar = ProgressBar(fund_list, 'Allocations Cal Start')  # 使用ProgressBar记录并循环进度
        holder = []
        # 使用multiprocessing模块进行运算仅可以在console下使用，jupyter notebook或python console
        # 中会无法运行
        if multi:
            pool = Pool()

            # pool_holder = {}

            def log_result(result):
                fund_code, ind_holding_est = result
                holder.append(ind_holding_est)
                bar.show(fund_code)

            for fund in fund_list:
                pool.apply_async(self.single_fund_allocation,
                                 (fund, obs_months,),
                                 {'bar': bar},
                                 callback=log_result)
            pool.close()
            pool.join()

        else:
            for fund in fund_list:
                holder.append(self.single_fund_allocation(fund, obs_months, bar=bar))
        return pd.concat(holder, sort=True)

    def get_fund_neutralized_allocation(self, obs_months=3, neutralized_months=24, renew=False, **kwargs):
        """
        获取中性化处理后的基金仓位信号，如果本地不存在则重新计算并保存本地
        :param obs_months: int, default 3
            以过去多久时间（多少个月）的数据来进行估算基金行业配合仓位，默认为3个月
        :param neutralized_months: int, default 24
            以过去多久时间（多少个月）的仓位数据进行中性化处理，默认为24个月
        :param renew: bool, default False
            是否重新计算，默认否
        :param kwargs: dict
            其他用于get_fund_allocation_est的参数
        :return: pandas DataFrame
            返回基金行中性化处理后的基金仓位信号，结构如下

                   tdate   sw_code    signal  fund_code
            0 2015-06-30  110000.0  0.639269  000056.OF
            1 2015-06-30  210000.0  0.497937  000056.OF
            2 2015-06-30  220000.0  0.010903  000056.OF
            3 2015-06-30  230000.0  0.616690  000056.OF
            4 2015-06-30  240000.0  0.070623  000056.OF

        """
        file_dir = join(self._data_path, 'neutralized_allocation_est_{}_{}.csv'.format(obs_months, neutralized_months))
        if exists(file_dir) and not renew:
            return pd.read_csv(file_dir, index_col=0, parse_dates=['tdate'])
        else:
            data = self.call_all_fund_neutralization(obs_months=obs_months,
                                                     neutralized_months=neutralized_months,
                                                     **kwargs)
            data.to_csv(file_dir)
            return data

    def call_all_fund_neutralization(self, obs_months=3, neutralized_months=24, **kwargs):
        """
        计算中性化处理后的基金仓位信号
        :param obs_months: int, default 3
            以过去多久时间（多少个月）的数据来进行估算基金行业配合仓位，默认为3个月
        :param neutralized_months: int, default 24
            以过去多久时间（多少个月）的仓位数据进行中性化处理，默认为24个月
        :param kwargs: dict
            其他用于get_fund_allocation_est的参数
        :return: pandas DataFrame
            返回基金行中性化处理后的基金仓位信号，结构如下

                   tdate   sw_code    signal  fund_code
            0 2015-06-30  110000.0  0.639269  000056.OF
            1 2015-06-30  210000.0  0.497937  000056.OF
            2 2015-06-30  220000.0  0.010903  000056.OF
            3 2015-06-30  230000.0  0.616690  000056.OF
            4 2015-06-30  240000.0  0.070623  000056.OF

        """
        all_allocations_est = self.get_fund_allocation_est(obs_months=obs_months, **kwargs)
        fund_list = all_allocations_est['fund_code'].unique().tolist()
        bar = ProgressBar(fund_list, 'Neutralization Cal Start')
        holder = []
        for fund in fund_list:
            bar.show(fund, False)
            holder.append(self.single_fund_neutralization(fund, all_allocations_est, neutralized_months))
            bar.show(fund, True)
        return pd.concat(holder)

    def single_fund_neutralization(self, fund_code, uni_allocations, neutralized_months=24):
        """
        计算单只基金中性化处理后的基金仓位信号
        1. 首先计算基金的净行业配置比例，即w(基金j行业i净配置) = w(基金j行业i配置估算) - w(市场行业i的市值权重)
        2. 在时间序列上计算每个行业的中心化配置中心化信号，即

            s(t) = [w(t) - min(w(T))]/[max(w(T)) - min(w(T))]

            s(t): t时刻的配置信号
            w(t): t时刻的权重估算
            w(T): 权重估算的历史时间序列
        :param fund_code: str
            基金代码，需为本地基金数据中的列名
        :param uni_allocations: pandas DataFrame
            全部基金的仓位估算，为get_fund_allocation_est返回的结果，结构如下

             allocation_est  fund_code  sw_code      tdate
            0          0.0334  000056.OF   110000 2014-07-31
            1          0.0253  000056.OF   210000 2014-07-31
            2          0.0245  000056.OF   220000 2014-07-31
            3          0.0420  000056.OF   230000 2014-07-31
            4          0.0237  000056.OF   240000 2014-07-31

        :param neutralized_months: int, default 24
            以过去多久时间（多少个月）的仓位数据进行中性化处理，默认为24个月
        :return: pandas DataFrame
            返回基金行中性化处理后的基金仓位信号，结构如下

                   tdate   sw_code    signal  fund_code
            0 2015-06-30  110000.0  0.639269  000056.OF
            1 2015-06-30  210000.0  0.497937  000056.OF
            2 2015-06-30  220000.0  0.010903  000056.OF
            3 2015-06-30  230000.0  0.616690  000056.OF
            4 2015-06-30  240000.0  0.070623  000056.OF

        """
        selected_allocation = uni_allocations[uni_allocations['fund_code'] == fund_code]. \
            pivot('tdate', 'sw_code', 'allocation_est')
        mkt_ind_allocation = self._mkt_ind_weight.reindex(index=selected_allocation.index,
                                                          columns=selected_allocation.columns).dropna()
        # 将估算的行业配置权重减去市场行业市值权重得到行业配置净权重
        net_allocation = selected_allocation - mkt_ind_allocation * selected_allocation.sum()
        # 使用DataFrame的rolling+apply方法对净权重的时间序列进行中性化处理，其中rolling.apply的元素
        # 为numpy ndarray，故需要使用numpy的方法
        neutralized = net_allocation.rolling(neutralized_months, neutralized_months).apply(
            lambda arr: (arr[-1] - arr.min()) / (arr.max() - arr.min()),
            raw=True).dropna().stack().reset_index().rename(
            columns={'level_0': 'tdate', 'level_1': 'sw_ind', 0: 'signal'})
        neutralized['fund_code'] = fund_code
        return neutralized

    def cal_signals(self, neutralized_allocations=None, bkw_months=1, sig_months=24, renew=False):
        """
        计算基金的轮动能力信号，步骤如下
        1. 计算中性化后的基金轮动信号，与当期基各行业指数收益的秩相关性，采用
        pandas.DataFrame.corr(method='spearman')
        2. 计算基金秩相关性时间序列的t统计量，即作为基金轮动能力信号，
        其中，Null Hypothesis: =0

        :param neutralized_allocations: pandas DataFrame, default None
            全部基金的中心化配置信号，即get_fund_neutralized_allocation返回结果，
            如入参值为None，则将执行get_fund_neutralized_allocation
            结构如下

                   tdate   sw_code    signal  fund_code
            0 2015-06-30  110000.0  0.639269  000056.OF
            1 2015-06-30  210000.0  0.497937  000056.OF
            2 2015-06-30  220000.0  0.010903  000056.OF
            3 2015-06-30  230000.0  0.616690  000056.OF
            4 2015-06-30  240000.0  0.070623  000056.OF

        :param bkw_months: int, default 1
            以多久窗口（多少月）计算行业指数的收益，默认为1个月
        :param sig_months: int, default 24
            考察多久（多少月）的秩相关性序列并计算t统计量，默认为24个月
        :param renew: bool, default False
            是否重新计算，默认否
        :return: pandas DataFrame
            返回基金的轮动能力信号指标，结构如下

               fund_code      tdate   signals
            0  000001.OF 2009-11-30 -3.825807
            1  000001.OF 2009-12-31 -3.923274
            2  000001.OF 2010-01-29 -3.902433
            3  000001.OF 2010-02-26 -4.350009
            4  000001.OF 2010-03-31 -4.437821

        """
        file_dir = join(self._data_path, '{}_{}_signals.csv'.format(bkw_months, sig_months))
        if exists(file_dir) and not renew:
            signals = pd.read_csv(file_dir, parse_dates=['tdate'], index_col=0)
            return signals

        holder = {}
        if neutralized_allocations is None:
            neutralized_allocations = self.get_fund_neutralized_allocation()
        group_df = neutralized_allocations.groupby(['tdate', 'fund_code'])
        bar = ProgressBar(group_df, 'Processing Signal')
        # 对按照每个时间点和每个基金对中性化后的配置信号进行分组
        for (tdate, fund_code), neutralized in group_df:
            bar.show('{} {}'.format(tdate.strftime('%Y-%m-%d'), fund_code), False)
            # 获得在该时间点过去bkw_months个月的所有行业指数的累计收益
            ind_return = ((self._industrial_indices.loc[
                           (pd.to_datetime(tdate) - pd.offsets.MonthBegin(bkw_months)):pd.to_datetime(tdate),
                           :] + 1).prod() - 1).to_frame().reset_index().rename(columns={0: 'bkwd_return'})
            # 计算中性化后的行业配置权重与行业指数的过去累计收益的秩相关性，这里使用DataFrame.corr(method='spearman')
            # 得到基金行业配置能力raw_signal
            holder.update({(tdate, fund_code): neutralized.merge(ind_return, on='sw_code')[['signal', 'bkwd_return']].
                          corr(method='spearman').iloc[0, 1]})
            bar.show('{} {}'.format(tdate.strftime('%Y-%m-%d'), fund_code), True)
        raw_signals = pd.Series(holder).to_frame().reset_index().rename(
            columns={'level_0': 'tdate', 'level_1': 'fund_code', 0: 'signals'})
        # 需要对每只基金的行业配置能力raw_signal在sig_months个月内的值进行t检验，将t-stats的值作为最终基金的行业
        # 轮动能力的信号（指标）
        fund_signals = raw_signals.groupby('fund_code'). \
            apply(lambda df: df.set_index('tdate').sort_index()['signals'].
                  rolling(sig_months, sig_months).apply(self.signal_ttest, raw='True')). \
            dropna().reset_index()
        fund_signals.to_csv(file_dir)
        return fund_signals

    @staticmethod
    def signal_ttest(signals, pop_mean=0):
        """
        计算信号序列t统计量
        :param signals: 1-D alike datatype
            信号序列
        :param pop_mean: int, default 0
            全样本均值（或Null Hypothesis）
        :return: float
            t统计量
        """
        t, _ = ttest_1samp(signals, pop_mean)
        return t

    def rank_ic_cal(self, fund_signals=None):
        """
        计信号不同lags的IC统计
        :param fund_signals: pandas DataFrame
            cal_signals的返回结果，结构如下

               fund_code      tdate   signals
            0  000001.OF 2009-11-30 -3.825807
            1  000001.OF 2009-12-31 -3.923274
            2  000001.OF 2010-01-29 -3.902433
            3  000001.OF 2010-02-26 -4.350009
            4  000001.OF 2010-03-31 -4.437821

        :return: tuple of dicts
            按以下顺序返回两个dict
            1. raw_ic序列
            2. 不同lag对应的IC_mean
        """

        if fund_signals is None:
            fund_signals = self.cal_signals()
        ic_mean_holder = {}
        ic_holder = {}
        all_fund_fwd = self._fund_fwd
        for i in range(1, 13):
            fund_fwd = all_fund_fwd[all_fund_fwd['fwd_window'] == i]
            rank_ic = fund_signals.merge(fund_fwd, on=['tdate', 'fund_code']).groupby('tdate')[
                ['signals', 'fwd_return']].apply(lambda df: df.corr(method='spearman').iloc[0, 1])
            ic_holder.update({'{}_lags'.format(i): rank_ic})
            ic_mean_holder.update({'{}_lags'.format(i): rank_ic.mean()})
        return ic_holder, ic_mean_holder

    def plt_bar_series(self, series):
        """
        对pandas Series做bar plot，主要为做不同lag的IC_means
        :param series: pandas
        :return: None
            无输出结果，仅将图片保存为~/dinosaur/intermediate_results/IC_lags.png
        """
        assert isinstance(series, pd.Series)
        fig, axs = plt.subplots(figsize=(10, 5))
        axs.bar(series.keys(), series.values, width=0.5)
        fig.suptitle('IC_mean vs different lags.')
        fig.savefig(join(self._data_path, pardir, 'intermediate_results', 'IC_lags.png'))
        # print()

    def simple_back_testing(self, fund_signals=None, groups=5, ratio=1.):
        """
        依据筛选基金信号构造简单回测，并以等权组合作为benchmark同样进行回测
        :param fund_signals: pandas DataFrame
            为cal_signals的返回结果，结构如下

               fund_code      tdate   signals
            0  000001.OF 2009-11-30 -3.825807
            1  000001.OF 2009-12-31 -3.923274
            2  000001.OF 2010-01-29 -3.902433
            3  000001.OF 2010-02-26 -4.350009
            4  000001.OF 2010-03-31 -4.437821

        :param groups: int, default 5
            每次调仓按多少组进行分组，默认为5组
        :param ratio: float, default 1.
            每期根据降序signal的前多少构成分组组合，默认1，即不进行筛选
        :return: None
            将各个组合的净值曲线保存在~/indcyle/intermediate_results/FundCyclingRankingNAV.png
            将各个组合的统计指标保存在~/indcyle/intermediate_results/FundCyclingRankingSummary.csv
        """
        if fund_signals is None:
            fund_signals = self.cal_signals()
        section_cum_return = np.ones(6)
        cum_holder = []
        group_holder = fund_signals.groupby('tdate')
        bar = ProgressBar(group_holder, 'Run Backtesting')
        for tdate, df in group_holder:
            bar.show(tdate.strftime('%Y-%m-%d'), False)
            # 在该时间点获取基金行业轮动能力指标并排序
            whole_slice = df.sort_values('signals', ascending=False).set_index('fund_code')['signals']
            # 按照比例挑选在当前时间点前ratio的基金作为回测基金池
            selected_slice = whole_slice.iloc[:int(whole_slice.shape[0] * ratio // 1)]
            # 对基金池内基金按照指标进行分组
            labels = pd.qcut(selected_slice, q=10, labels=False)
            # 在fund_return中找到tdate的下一个时间戳
            inception_date = min(self._fund_return.index[self._fund_return.index > tdate])
            # 按照rebanlce频率，找到下次换仓前的结束时间
            end_date = max(
                self._fund_return.index[self._fund_return.index <= inception_date + pd.offsets.MonthEnd(1)])
            # 制作mask DataFrame用于分组
            mask = pd.DataFrame(index=self._fund_return.loc[inception_date:end_date, :].index,
                                columns=self._fund_return.columns).fillna(labels)
            # 所有基金在这个时间段内的日频收益
            returns = self._fund_return.loc[inception_date:end_date, mask.columns]
            # 所有基金在这个时间段内的日频累计收益
            cum_return = (1 + returns).cumprod()
            cum_return /= cum_return.iloc[0, :]
            section_holder = []
            # 对每个分组进行的基金，将其在这段时间内的累计收益进行平均，即得到在期初等权持有该组基金的组合的累计净值
            for i in range(groups):
                section_holder.append(cum_return.where(mask.isin([i * 2, i * 2 + 1])).mean(axis=1).
                                      rename('group_{}'.format(i + 1)))
            section_holder.append(cum_return.mean(axis=1).rename('equal_weight'))
            # 将所有组的累计净值延axis=1拼接并乘以上一段时间所有组累计收益的期末值，如果为第一期则全为1
            group_section_cum = pd.concat(section_holder, axis=1) * section_cum_return
            section_cum_return = group_section_cum.iloc[-1, :].values
            # 将该时间段内的累计净值与之前已经拼接好的累计净值进行拼接
            cum_holder.append(group_section_cum)
            bar.show(tdate.strftime('%Y-%m-%d'), True)
        nav = pd.concat(cum_holder)
        summary = SummaryStatistic(nav).summary  # 计算各个组净值的统计分析指标
        # 将结果输出并保存
        print(summary)
        summary.to_csv(join(self._data_path, pardir, 'intermediate_results', 'FundCyclingRankingSummary.csv'))
        nav.plot()
        plt.savefig(join(self._data_path, pardir, 'intermediate_results', 'FundCyclingRankingNAV.png'))
        # nav.to_clipboard()

    @staticmethod
    def fwd_return(windows, obs_dates, obs_returns):
        """
        计算对应收益序列在各个时间点的下期收益，主要用于计算IC相关指标，
        计算量相对较大，需要预计算，即为生成~/indcyle/data/fund_fwd_return.csv与
        ~/dinosaur/data/ind_fwd_return.csv的方法
        :param windows: iter of int
            未来收益的窗口，如range(1, 13)，会分别生成在每个观测时间点向前1至12个月
            的未来收益
        :param obs_dates: iter of timestamp, or timestamp like str
            观测时间的iter，如['2017-01-31', '20170228', '2017-3-31']，会返回所有
            在obs_returns中的ticker在'2017-01-31', '2017-02-28', '2017-03-31'上，
            所有windows的forward return
        :param obs_returns: pandas DataFrame
            原始收益序列的DataFrame，为self._fund_return或self._industrial_indices的
            形式，格式如下

            fund_code   000001.OF  000011.OF    ...      960028.OF  960033.OF
            tdate                               ...
            2006-01-04        NaN        NaN    ...            NaN        NaN
            2006-01-05   0.016407   0.014691    ...            NaN        NaN
            2006-01-06   0.012107   0.006757    ...            NaN        NaN
            2006-01-09   0.003013   0.010547    ...            NaN        NaN
            2006-01-10   0.003975   0.010436    ...            NaN        NaN

        :return: pandas DataFrame
            未来收益序列，格式如下

                   tdate  fund_code  fwd_return  fwd_window
            0 2008-12-31  000001.OF    0.066619           1
            1 2008-12-31  000011.OF    0.132777           1
            2 2008-12-31  000021.OF    0.049477           1
            3 2008-12-31  000031.OF    0.083067           1
            4 2008-12-31  002011.OF    0.062312           1

        """
        fwd_return_holder = []
        reference = obs_returns.index
        latest = reference[-1]
        assert obs_returns.index.name is not None
        assert obs_returns.columns.name is not None
        index_name = obs_returns.index.name
        col_name = obs_returns.columns.name
        for window in windows:
            holder = {}
            for date in obs_dates:
                element = pd.to_datetime(date)
                next_inception_date = min(reference[element <= reference])
                if next_inception_date + pd.offsets.MonthEnd(window) >= latest:
                    continue
                termination_date = max(reference[next_inception_date + pd.offsets.MonthEnd(window) >= reference])
                holder.update(
                    {element: (obs_returns.loc[next_inception_date:termination_date, :] + 1).prod(axis=0,
                                                                                                  skipna=False) - 1})
            fwd_return = pd.DataFrame(holder).T.stack().reset_index().rename(
                columns={'level_0': index_name, 'level_1': col_name, 0: 'fwd_return'})
            fwd_return['fwd_window'] = window
            fwd_return_holder.append(fwd_return)
        return pd.concat(fwd_return_holder)

    def ic_analysis(self, ic_ser, name, plot=False):
        """
        分析raw_ic序列，计算IC_mean、IC_IR, Annual_IR及IC的t-stats、p-values并将结果存储在
        ~/dinosaur/intermediate_results/{name}.png

        :param ic_ser: pandas Series
            raw_ic序列，为rank_ic_cal返回的ic_holder中的元素，形式如下

            tdate
            2009-11-30    0.016325
            2009-12-31    0.303952
            2010-01-29    0.106731
            2010-02-26    0.231541
            2010-03-31    0.335001

        :param name: str
            图片名字
        :param plot: bool, default False
            是否画图，
            如果画图则将IC序列做bar plot并保存在
            ~/dinosaur/intermediate_results/{name}.png
            如果不画图，则将计算IC_mean、IC_IR, Annual_IR及IC的t-stats、p-values
            打印
        :return: None
        """
        ic_ser = pd.Series(ic_ser)
        multi = (ic_ser.index[-1] - ic_ser.index[0]).days/365/ic_ser.shape[0]
        ic_mean = ic_ser.mean()
        ic_ir = ic_ser.mean() / ic_ser.std()
        annl_ir = ic_ir * (multi**0.5)
        t, p = ttest_1samp(ic_ser, 0)
        title = '{}\n({})'.format(
            '{}~{}'.format(ic_ser.index.min().strftime('%Y-%m-%d'), ic_ser.index.max().strftime('%Y-%m-%d')),
            'ic_mean = {:.2f}, ic_ir = {:.2f}, annl_ir = {:.2f}, t_stats = {:.2f}, p_values = {:.2f}'.
                format(ic_mean, ic_ir, annl_ir, t, p))
        if plot:
            fig, axs = plt.subplots(figsize=(15, 5))
            axs.bar(ic_ser.index, ic_ser.values, width=10)
            fig.suptitle(title)
            fig.savefig(join(self._data_path, pardir, 'intermediate_results', '{}.png'.format(name)))
        else:
            print(title)


if __name__ == '__main__':
    Data_path = 'D:\\Programming Platform\\Research\\IndustryCycle\\dinosaur\\data'
    research = IndustryCycling(Data_path)
    Allocations = research.get_fund_allocation_est(obs_months=1)  # obs_months=3, renew=True, multi=True)
    Neutralized = research.get_fund_neutralized_allocation(obs_months=1, neutralized_months=12)  # , renew=True)
    Signals = research.cal_signals(Neutralized)  # , renew=True)
    IC_holder, IC_mean_holder = research.rank_ic_cal(Signals)
    research.plt_bar_series(pd.Series(IC_mean_holder))
    research.ic_analysis(IC_holder['1_lags'], True)
    research.simple_back_testing(Signals)
    plt.show()
    # allocations = research.single_fund_allocation('000001.OF')
    # print(allocations)

    # iterItems = range(100000)
    # bar = ProgressBar(iterItems, 'Test')
    # for i in iterItems:
    #     bar.show(str(i), False)
    #     bar.show(str(i), True

    # yy = np.random.randn(400)
    # xx = np.random.randn(400, 20)
    # opt = IndustryCycling.qp_estimate_exposure(yy, xx)
    # print(opt)
    # print(opt[1:].sum())
