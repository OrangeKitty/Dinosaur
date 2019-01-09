from os.path import join
from pandas import to_datetime, DataFrame, qcut, read_csv
from pandas.tseries.offsets import MonthEnd

from dinosaur.mod.interface import AbstractStrategy
from dinosaur.mod.environment import LocalDataEnv


class IndustrialIndicesBuyAndHold(AbstractStrategy):
    """
    该策略为一个Buy&Hold的测试策略
    environment为申万策略指数，即industrial_indices.csv文件的位置
    策略逻辑：
    1. 分别建立两'a'与'b'两个账户，
    2. 两个账户在执行交易的第一天分别买入农林牧渔指数（110000）与
    采掘指数（210000）
    3. 每个tick，将该账户下的全部持仓卖掉再全仓买入该ticker

    目的为了检验回测模块的正确运行

    """

    def __init__(self, data_path):
        super().__init__(LocalDataEnv(data_path))
        self.first_tick = None

    def init(self):
        self.register_account('a')
        self.register_account('b')
        self.first_tick = True

    def handle_bar(self):
        if self.first_tick:
            self.order_by_weight(110000, 'a', 1.)
            self.order_by_weight(210000, 'b', 1.)
            self.first_tick = False

        if to_datetime(self.cur_tick).month in (3, 6, 9, 12):
            self.order_by_weight(110000, 'a', 0., to_target=True)
            self.order_by_weight(210000, 'b', 0., to_target=True)
            self.order_by_weight(110000, 'a', 1., to_target=True)
            self.order_by_weight(210000, 'b', 1., to_target=True)

    @property
    def turn_over(self):
        holder = {}
        for name, account in self._accounts_holder.items():
            holder.update({name: account.turnover_log/2})
        return DataFrame(holder)


class FundSignalTesting(AbstractStrategy):
    """
    基金行业轮动能力指标策略回测

    策略准备
    1. environment为基金净值的LocalDataEnv，数据位置
        ~/dinosaur/data/fund_nav.csv文
    2. 将基金轮动能力信号读入策略中作为辅助，文件位置
        ~/dinosaur/intermediate_results/1_24_signals.csv

    策略逻辑
    1. init中建立5个分组账户，命名为group_1~5，并读取1_24_signals.csv中的行业轮动能力指标
        作为下单信号参考
    2. handler_bar中判断是否为第一个回测日或月末，如果是
        2.1 通过get_signals获取该交易tick上的所有基金的行业轮动能力指标
        2.2 对指标升序排列后等分5组，分别在group_1~5的账户下对这5组基金使用order_equal_weights
            进行等权下单
        2.3 每月末调仓

    """

    def __init__(self, data_path):
        env = LocalDataEnv(join(data_path, 'fund_nav.csv'))
        super().__init__(env)
        self.first_tick = None
        self.signals = read_csv(join(data_path, '1_24_signals.csv'), parse_dates=['tdate'], index_col=0)

    def init(self):
        for i in range(5):
            self.register_account('group_{}'.format(i + 1))
        self.first_tick = True

    def handle_bar(self):
        if self.first_tick:
            signals = self.get_signals()
            ranking = qcut(signals, 5, labels=False)
            for i in range(5):
                self.order_equal_weights(ranking[ranking == i].index, 'group_{}'.format(i + 1))
            self.first_tick = False

        if self.is_month_end:
            signals = self.get_signals()
            ranking = qcut(signals, 5, labels=False)
            for i in range(5):
                self.order_equal_weights(ranking[ranking == i].index, 'group_{}'.format(i + 1))

    def get_signals(self):
        # 获得当天的所有基金的轮动能力指标
        nearest_date = max(self.signals['tdate'][self.signals['tdate'] <= to_datetime(self.cur_tick)])
        return self.signals[self.signals['tdate'] == nearest_date.strftime('%Y-%m-%d')][['fund_code', 'signals']].\
            set_index('fund_code')['signals']

    def order_equal_weights(self, ticker_list, account_name):
        # 在account_name下对ticker_list中的每个ticker进行等权重下单操作
        # 如果ticker存在于之前的holding中，将其仓位比率调整至目标权重
        w = 1/ticker_list.__len__()
        cur_holding = self._accounts_holder[account_name].cur_holding
        for ticker, _ in cur_holding.items():
            if ticker not in ticker_list:
                self.order_by_weight(ticker, account_name, 0, True)

        for ticker in ticker_list:
            self.order_by_weight(ticker, account_name, w, True)

    @property
    def turn_over(self):
        # 各个账户的还手率历史
        holder = {}
        for name, account in self._accounts_holder.items():
            holder.update({name: account.turnover_log / 2})
        return DataFrame(holder)


class IndustryCyclingStrategy(AbstractStrategy):
    """
    基金行业轮动能力行业配置策略回测

    策略准备
    environment为基金净值的LocalDataEnv，数据位置industrial_indices.csv文件的位置
    同时，需要将以下数据先读入到策略中作为策略辅助，以下数据均为通过indcycle.research.\
    analysis中的IndustryCycling模块预计算得来或直接在excel中处理得来
        1. 以过去三个月未窗口的日度数据估算基金的行业配置比例
            ~/dinosaur/intermediate_results/fund_allocation_est_3.csv
        2. 全市场行业市值权重
            ~/dinosaur/data/ind_weight.csv
        3. 基金行业轮动能力指标（如研报中所述，采用的是1个月窗口估算仓位，24个月窗口进
            行中性化处理）
            ~/dinosaur/intermediate_results/1_24_signals.csv

    策略逻辑
    1. init中建立3个分组账户，分别命名为top，bottom，equal_weight，分别对应于行业轮
        转信号排名前10%的行业，排名后10%的行业，以及等权行业组合
    2. handler_bar中判断是否为第一个回测日或月末，如果是
        2.1 判断是否为策略开始时间或季度末，是则执行策略
        2.2 首先读取当前tick的所有基金轮动能力指标，选取前50%作为新的基金池
        2.3 将基金池中所有基金的行业仓位估计（3个月日度净值估算）按照各个行业取平均值，
            得到该时间点基金池的各个行业的配置信号，减去该时间点上市场的行业配置比率，
            得到基金池的行业净配置仓位信号
        2.4 通过回溯2年（8个季度）的行业仓位净配置仓位信号，按照时间序列中性化的方法求
            的该时间点的各个行业的配置信号，降序排列，选取前10%的行业作为top账户下的
            等权下单操作对象，后10%作为bottom账户下的等权下单操作，因为一共只有29个申
            万行业，故策略中直接取top3以及bottom3行业进行相应操作
        2.5 在equal_weight账户中对全部行业进行等权下单操作
        2.3 每季度末调仓

    """

    def __init__(self, data_path):
        env = LocalDataEnv(join(data_path, 'industrial_index.csv'),
                           is_nav=False)
        super().__init__(env)
        self.first_tick = None
        self.neutralized_holder = {}
        self.allocation_est = read_csv(
            join(data_path, 'fund_allocation_est_3.csv'),
            parse_dates=['tdate'],
            index_col=0
        )
        self.mkt_ind_allocation = read_csv(
            join(data_path, 'ind_weight.csv'),
            parse_dates=['tdate']
        ).set_index('tdate')
        self.mkt_ind_allocation.columns = [int(x) for x in self.mkt_ind_allocation.columns]
        self.signals = read_csv(
            join(data_path, '1_24_signals.csv'),
            parse_dates=['tdate'],
            index_col=0
        )

    def init(self):
        self.register_account('top')
        self.register_account('bottom')
        self.register_account('equal_weight')
        self.first_tick = True

    def handle_bar(self):
        if self.first_tick:
            signals = self.get_signals()
            ranking = signals.sort_values()
            self.order_equal_weights(ranking.head(3).index, 'bottom')
            self.order_equal_weights(ranking.tail(3).index, 'top')
            self.order_equal_weights(ranking.index, 'equal_weight')

        if self.is_quarter_end and not self.first_tick:
            signals = self.get_signals()
            ranking = signals.sort_values()
            self.order_equal_weights(ranking.head(3).index, 'bottom')
            self.order_equal_weights(ranking.tail(3).index, 'top')
            self.order_equal_weights(ranking.index, 'equal_weight')

        self.first_tick = False

    def order_equal_weights(self, ticker_list, account_name):
        # 在account_name下对ticker_list中的每个ticker进行等权重下单操作
        # 如果ticker存在于之前的holding中，将其仓位比率调整至目标权重
        w = 1/ticker_list.__len__()
        cur_holding = self._accounts_holder[account_name].cur_holding
        for ticker, _ in cur_holding.items():
            if ticker not in ticker_list:
                self.order_by_weight(ticker, account_name, 0, True)

        for ticker in ticker_list:
            self.order_by_weight(ticker, account_name, w, True)

    def get_signals(self):
        # 获得当前时间点的行业配置信号
        obs_date = to_datetime(self.cur_tick)
        for i in range(1, 9):
            if obs_date in self.neutralized_holder:
                continue
            else:
                self.update_neutralized_holder(obs_date)
            obs_date = max(filter(lambda x: x <= obs_date - MonthEnd(3), self.time_axis))
        signals = DataFrame(self.neutralized_holder).T.sort_index().loc[:to_datetime(self.cur_tick), :].tail(8).\
            apply(lambda ser: (ser.iloc[-1] - ser.min())/(ser.max() - ser.min()))
        return signals

    def update_neutralized_holder(self, tick, ratio=0.5):
        # neutralized_holder记录了每个季度末整个基金池的净仓位
        # update_neutralized_holder则在新的季末tick按照基金轮动能力信重新构建基金池，计算
        # 基金池的经仓位，并加入到neutralized_holder中
        raw_signals = self.signals[self.signals['tdate'] == tick][['fund_code', 'signals']].\
            set_index('fund_code')['signals']
        pool = list(raw_signals.sort_values(ascending=False).iloc[:int(ratio*raw_signals.shape[0])].index)
        raw_allocation = self.allocation_est[self.allocation_est['tdate'] == tick]
        filtered_allocation = raw_allocation[raw_allocation['fund_code'].isin(pool)]
        avg_ind_allocation = filtered_allocation.groupby('sw_code')['allocation_est'].mean()
        mkt_ind_allocation = self.mkt_ind_allocation.loc[self.cur_tick, :]

        self.neutralized_holder.update({tick: avg_ind_allocation - mkt_ind_allocation})

    @property
    def nav(self):
        # 各个账户的净值
        holder = {}
        for name, account in self._accounts_holder.items():
            holder.update({name: account.account_log})
        return DataFrame(holder)

    @property
    def turn_over(self):
        # 各个账户的换手率历史
        holder = {}
        for name, account in self._accounts_holder.items():
            holder.update({name: account.turnover_log / 2})
        return DataFrame(holder)
