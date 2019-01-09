from abc import ABCMeta, abstractmethod
from pandas import DataFrame

from dinosaur.utils import ProgressBar, SummaryStatistic
from dinosaur.backtesting.brokerage import Account


class AbstractEnvironment(metaclass=ABCMeta):
    """
    Environment接口
    该类作为回测模块中使用的environment模块的抽象接口类，声明了必须要复写或声明的接口方法
    以下将在每个接口方法中进一步解释
    """

    @abstractmethod
    def get_bar(self, tick, ticker, counts=1):
        """
        获取数据bar接口
        :param tick: str, datetime datetime or pandas Timestamp
            下单时间，可以转换为时间戳的时间标识
        :param ticker: str or object
            下单对象，对象字符串标识或其他对象
        :param counts: int, default 1
            获取以tick时间为准向前多少条bar数据
        :return: list
            以正序获取counts个bar的数据
        """
        raise NotImplementedError

    @abstractmethod
    def get_time_axis(self):
        """
        获取environment内的时间轴
        运行策略时，会按照时间轴上每一个时间戳进行循环
        :return: list
            按升序排列的时间戳
        """
        raise NotImplementedError

    @abstractmethod
    def get_range_time_axis(self, start=None, end=None):
        """
        按开始日期与结束日期获得时间戳序列
        在策略运行时调用该模块
        :param start: str, datetime datetime or pandas Timestamp
            开始时间，可以转换为时间戳的时间标识
        :param end: str, datetime datetime or pandas Timestamp
            结束时间，可以转换为时间戳的时间标识
        :return: list
            按升序排列的时间戳
        """
        raise NotImplementedError

    @abstractmethod
    def get_avail_ticker(self, tick):
        """
        获取目标tick上有效的所有ticker的标识
        :param tick: str, datetime datetime or pandas Timestamp
            目标tick，可以转换为时间戳的时间标识
        :return: list
            一列ticker的标识
        """
        raise NotImplementedError

    @abstractmethod
    def is_week_end(self, tick):
        """
        目标tick是否为周末
        :param tick: tr, datetime datetime or pandas Timestamp
            目标tick，可以转换为时间戳的时间标识
        :return: bool
        """
        raise NotImplementedError

    @abstractmethod
    def is_month_end(self, tick):
        """
        目标tick是否为月末
        :param tick: tr, datetime datetime or pandas Timestamp
            目标tick，可以转换为时间戳的时间标识
        :return: bool
        """
        raise NotImplementedError

    @abstractmethod
    def is_quarter_end(self, tick):
        """
        目标tick是否为季末
        :param tick: tr, datetime datetime or pandas Timestamp
            目标tick，可以转换为时间戳的时间标识
        :return: bool
        """
        raise NotImplementedError

    @abstractmethod
    def is_year_end(self, tick):
        """
        目标tick是否为年末
        :param tick: tr, datetime datetime or pandas Timestamp
            目标tick，可以转换为时间戳的时间标识
        :return: bool
        """
        raise NotImplementedError


class AbstractStrategy(metaclass=ABCMeta):
    """
    Strategy接口
    该类作为所有策略的接口抽象类，
    所有新策略均需继承该类并复写或重新定义该类中的接口方法
    策略逻辑需要在 init与handle_bar中进行实现，其中
        init仅在loop每个environment的tick前执行一次；
        handle_bar则会在每个tick loop中执行一次，默认为每个tick更新后执行，
        且下单位在每个tick的close后全部按close price进行交割
    """

    def __init__(self, env):
        """
        模块初始化
        如果新策略需要定义自己的__init__，则需要在其中执行 super().__init__(env)
        :param env: instance of AbstractEnvironment or subclass of it
            数据环境，必须为AbstractEnvironment或其子类生成的实例对象
        """
        assert isinstance(env, AbstractEnvironment)
        self._env = env
        self._accounts_holder = {}
        self._cur_tick = None

    @abstractmethod
    def init(self):
        """
        策略循环tick之前执行，且运行中只执行一次
        建议在该方法按照回测需求注册账号
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def handle_bar(self):
        """
        策略的核心逻辑
        每循环一个tick则执行一次，会在对所有账户中的持仓进行更新后再执行
        :return:
        """
        raise NotImplementedError

    def register_account(self, account_name, init_invest=100, fee_rate=0.):
        """
        注册账户
        :param account_name: str
            账户名称
        :param init_invest: float, default 100.
            账户初始投资金额
        :param fee_rate: float, default 0.
            账户交易费用率假设（单边）,默认为0
        :return: None
        """
        self._accounts_holder.update({account_name: Account(init_invest, self._env, fee_rate)})

    def order_by_weight(self, ticker, account_name, order_weight, to_target=False):
        """
        按照账户市值的响应比率下单
        :param ticker: str or object
            下单ticker的标识，字符串或其他对象
        :param account_name: str
            操作目标账户的名称
        :param order_weight: float
            下单比率（权重）
        :param to_target: bool, False
            是否按照下单比率（权重）对目标账户的持仓比率进行调整
        :return: None
        """
        self._accounts_holder.get(account_name).order_by_weight(self.cur_tick, ticker, order_weight, to_target)

    @property
    def cur_tick(self):
        """
        当前tick的str
        :return: str
        """
        return self._cur_tick.strftime('%Y-%m-%d')

    @property
    def is_month_end(self):
        """
        当前tick是否为月末
        :return: bool
        """
        return self._env.is_month_end(self.cur_tick)

    @property
    def is_quarter_end(self):
        """
        当前tick是否为季末
        :return: bool
        """
        return self._env.is_quarter_end(self.cur_tick)

    @property
    def time_axis(self):
        """
        回测时间轴
        :return: list
        """
        return self._env.get_time_axis()

    def run(self, start=None, end=None):
        """
        按照开始结束时间运行回测
        :param start: str, datetime datetime or pandas Timestamp
            开始时间，可以转换为时间戳的时间标识
        :param end: str, datetime datetime or pandas Timestamp
            结束时间，可以转换为时间戳的时间标识
        :return: list
            按升序排列的时间戳
        """
        self.init()
        tick_lib = self._env.get_range_time_axis(start, end)
        bar = ProgressBar(tick_lib, '{} Starts'.format(type(self).__name__), )
        for tick in tick_lib:
            bar.show('{}'.format(tick.strftime('%Y-%m-%d')), False)
            self._cur_tick = tick
            for _, account in self._accounts_holder.items():
                account.pre_action_update(tick)
            self.handle_bar()
            for _, account in self._accounts_holder.items():
                account.post_action_update(tick)
            bar.show('{}'.format(tick.strftime('%Y-%m-%d')), True)

    @property
    def nav(self):
        """
        所有账户的净值序列
        :return: pandas DataFrame
        """
        holder = {}
        for name, account in self._accounts_holder.items():
            holder.update({name: account.account_log})
        return DataFrame(holder)

    @property
    def summary(self):
        """
        所有账户的净值评估
        即为 incycle.utils.stats_lib中SummaryStatistic.summary的输出结果
        :return: pandas DataFrame
        """
        return SummaryStatistic(self.nav).summary
