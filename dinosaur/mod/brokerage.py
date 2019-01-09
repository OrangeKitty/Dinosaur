from collections import OrderedDict
from pandas import to_datetime, DataFrame, Series

# from dinosaur.backtesting.interface import AbstractEnvironment


class OrderElement(object):
    """
    模拟交易回测的底层模块——订单元素模块
    该类为每笔订单的元素模块，记录并制定每笔订单的行为，主要包括以下信息
    1. 属性
    1.1 下单对象（ticker）
    1.2 开（挂）单信息：时间，价格，股数（份额），开（挂）单金额，开（挂）单状态，开（挂）单方向
    1.3 成交信息：时间，价格，股数（份额），成交金额，成交状态，

    2. 方法
    2.1 成交
    2.2 关闭
    """

    def __init__(self, ticker, open_tick, open_price, open_share=None, open_amount=None):
        """
        初始化订单属性
        open_share和open_amount至少有一个不为空
        :param ticker: str or object
            下单对象
        :param open_tick: pandas Timestamp, datetime datetime or str
            开（挂）单时间
        :param open_price: float
            开（挂）单价格
        :param open_share: float, default None
            开（挂）单股数（份额）
        :param open_amount: float, default None
             开（挂）单金额
        """
        if open_share is None and open_amount is None:
            raise ValueError('At least one of "open_share" and "open_amount" not be None.')

        self.ticker = ticker
        self.open_tick = open_tick
        self.open_price = open_price
        self.open_share = open_amount/open_price if open_share is None else open_share
        self.open_amount = open_share*open_price if open_amount is None else open_amount
        self.open = True

        self.deal_tick = None
        self.deal_price = None
        self.deal_share = None
        self.deal_amount = None
        self.deal = False

        self.direction = 1 if self.open_share >= 0 else -1
        self.close = False

    def deal_order(self, deal_tick, deal_price, deal_share=None, deal_amount=None):
        """
        订单成交方法，按照相应参数完成订单交割
        deal_share和deal_amount至少有一个不为空
        :param deal_tick: pandas Timestamp, datetime datetime or str
            交割时间
        :param deal_price: float
            交割价格
        :param deal_share: float, default None
            交割股数（份额）
        :param deal_amount: float, default None
            交割金额
        :return: None
        """
        if deal_share is None and deal_amount is None:
            raise ValueError('At least one of "deal_share" and "deal_amount" not be None.')

        self.deal_tick = deal_tick
        self.deal_price = deal_price
        self.deal_share = deal_amount/deal_price if deal_share is None else deal_share
        self.deal_amount = deal_share*deal_price if deal_amount is None else deal_amount
        self.deal = True
        self.close = True
        self.open = False

    def close_order(self):
        """
        关闭订单
        :return: None
        """
        self.close = True
        self.deal = False
        self.open = False


class OrderBook(object):
    """
    模拟交易回测的底层模块——订单簿模块
    订单簿中用订单簿序号区分不同订单元素，订单元素即为OrderElement类生成的实例
    用OrderedDict按顺序记录每个交易行为，包括交割的订单、挂起的订单以及关闭的订单
    拥有以下方法：
    1. 开（挂）单
    2. 成交订单
    3. 获取所有开（挂）的订单
    4. 获取所有开（挂）的卖单
    5. 获取所有开（挂）的买单
    6. 获取对某个下单对象的所有开（挂）单
    """

    def __init__(self):
        self._book = OrderedDict()
        self._counter = 0
        pass

    def open_order(self, ticker, open_tick, open_price, open_share=None, open_amount=None):
        """
        开（挂）订单
        open_share和open_amount至少有一个不为空
        :param ticker: str or object
            下单对象
        :param open_tick: pandas Timestamp, datetime datetime or str
            开（挂）单时间
        :param open_price: float
            开（挂）单价格
        :param open_share: float, default None
            开（挂）单股数（份额）
        :param open_amount: float, default None
             开（挂）单金额
        """
        self._book.update({self._counter: OrderElement(ticker, open_tick, open_price, open_share, open_amount)})
        self._counter += 1

    def deal_order(self, order_id, deal_tick, deal_price, deal_share=None, deal_amount=None):
        """
        订单成交方法，按照相应参数完成订单交割
        deal_share和deal_amount至少有一个不为空
        :param order_id: int
            订单簿序号
        :param deal_tick: pandas Timestamp, datetime datetime or str
            交割时间
        :param deal_price: float
            交割价格
        :param deal_share: float, default None
            交割股数（份额）
        :param deal_amount: float, default None
            交割金额
        :return: None
        """
        self._book.get(order_id).deal_order(deal_tick, deal_price, deal_share, deal_amount)

    def get_open_orders(self):
        """
        获取所有开（挂）的订单
        :return: generator object
            以（订单簿序号，订单元素）为形式的generator
        """
        for _id, element in self._book.items():
            assert isinstance(element, OrderElement)
            if element.open:
                yield _id, element

    def sell_open_orders(self):
        """
        获取所有开（挂）的卖单
        :return: generator object
            以（订单簿序号，订单元素）为形式的generator
        """
        for _id, element in self._book.items():
            assert isinstance(element, OrderElement)
            if element.open and element.direction == -1:
                yield _id, element

    def buy_open_orders(self):
        """
        获取所有开（挂）的买单
        :return: generator object
            以（订单簿序号，订单元素）为形式的generator
        """
        for _id, element in self._book.items():
            assert isinstance(element, OrderElement)
            if element.open and element.direction == 1:
                yield _id, element

    def specific_open_orders(self, ticker):
        """
        获取对某个下单对象的所有开（挂）单
        :param ticker: str or object
            下单对象标识，字符串或其他对象
        :return: generator object
            以（订单簿序号，订单元素）为形式的generator
        """
        for _id, element in self._book.items():
            assert isinstance(element, OrderElement)
            if element.open and element.ticker == ticker:
                yield _id, element

    @property
    def view(self):
        """
        以pandas DataFrame的形式查看订单历史
        :return: pandas DataFrame
        """
        return DataFrame({_id: element.__dict__ for _id, element in self._book.items()}).T


class HoldingElement(object):
    """
    模拟交易回测的底层模块——持仓元素模块
    该类为每期的持仓元素模块，记录每个账户在每个时间点的持仓信息，并定义持仓行为，包括以下信息
    1. 属性
    1.1 持仓对象标识（ticker)
    1.2 持仓股数（份额）
    1.3 持仓金额
    1.4 持仓对象当前价格
    2. 方法
    2.1 更新持仓
    2.2 增加股数（份额）
    2.3 增加金额
    """

    def __init__(self, ticker, price, share=None, amount=None):
        """
        模块初始化
        share和amount至少有一个不为None
        :param ticker: str or object
            持仓对象标识，字符串或其他对象
        :param price: float
            持仓价格
        :param share:
            持仓股数（份额）
        :param amount:
            持仓金额
        """
        if share is None and amount is None:
            raise ValueError('At least one of "share" and "amount" should not be None.')
        self.ticker = ticker
        self.share = amount/price if share is None else share
        self.amount = share*price if amount is None else amount
        self.price = price

    def update(self, price, share_diff=None, amount_diff=None):
        """
        更新持仓元素
        单更新价格，则仅更新持仓金额，股数（份额）不变
        如果share_diff以及amount_diff中有一个有变化，表示有因为交易行为而导致的仓位变化
        :param price: float
            更新价格
        :param share_diff: float, default None
            股数（份额）增（减）量
        :param amount_diff: float, default None
            金额增（减）量
        :return: None
        """
        if share_diff is None and amount_diff is None:
            self.amount *= price/self.price
            self.price = price
        elif share_diff is None and amount_diff is not None:
            self.amount += amount_diff
            self.share += amount_diff/price
        elif share_diff is not None and amount_diff is None:
            self.amount += share_diff*price
            self.share += share_diff
        else:
            self.amount += amount_diff
            self.share += share_diff

    def add_share(self, share):
        """
        增（减）加持仓股数（份额）
        :param share: float
        :return: None
        """
        self.share += share
        self.amount += share*self.price

    def add_amount(self, amount):
        """
        增（减）加持仓金额
        :param amount: float
        :return: None
        """
        self.amount += amount
        self.share += amount/self.price


class HoldingBook(object):
    """
    模拟交易回测的底层模块——持仓簿模块
    持仓簿以每个持仓对象作区分持仓元素，持仓元素即为HoldingElement类生成的实例
    该类记录每个账户在每个时间点全部持仓信息，定义以下持仓簿行为方法
    1. 获得当前持仓信息
    2. 增加持仓
    3. 更新某个持仓元素
    4. 获得某个持仓元素的持股数或份额
    5. 获得某个持仓元素的持仓金额数
    6. 清楚无效持仓元素
    7. 返回当前持仓总金额
    """
    def __init__(self):
        self.cur_holding = {}
        # self._holding_history = {}
        pass

    # def update(self, tick):
        # pass

    def get_cur_holding(self):
        """
        获得当前持仓信息
        :return: dict
        """
        return self.cur_holding

    def add_holding(self, ticker, price, share=None, amount=None):
        """
        增加持仓
        share和amount至少有一个不为None
        :param ticker: str or object
            持仓对象，字符串或其他对象
        :param price: float
            持仓价格
        :param share: float, default=None
            持仓股数（份额）
        :param amount: float, default=None
            持仓金额
        :return: None
        """
        self.cur_holding.update({ticker: HoldingElement(ticker, price, share, amount)})

    def update_holding(self, ticker, price, share=None, amount=None):
        """
        更新持仓
        share和amount至少有一个不为None
        :param ticker: str or object
            持仓对象，字符串或其他对象
        :param price: float
            持仓价格
        :param share: float, default=None
            持仓股数（份额）
        :param amount: float, default=None
            持仓金额
        :return: None
        """
        if ticker not in self.cur_holding:
            self.add_holding(ticker, price, share, amount)
        else:
            self.cur_holding.get(ticker).update(price, share, amount)

    def get_amount(self, ticker):
        """
        获得某个持仓元素的持仓金额数
        :param ticker: str or object
            持仓对象，字符串或其他对象
        :return: float
        """
        if ticker in self.cur_holding:
            return self.cur_holding.get(ticker).amount

    def get_share(self, ticker):
        """
        获得某个持仓元素的持股数或份额
        :param ticker: str or object
            持仓对象，字符串或其他对象
        :return: float
        """
        if ticker in self.cur_holding:
            return self.cur_holding.get(ticker).share

    def flush(self):
        """
        清楚无效持仓元素
        将持仓股数（份额）为0或金额为0的持仓元素提出持仓簿
        :return: None
        """
        for ticker, element in self.cur_holding.items():
            assert isinstance(element, HoldingElement)
            if round(element.share, 6) == 0. or round(element.amount, 6) == 0.:
                self.cur_holding.pop(ticker)

    @property
    def total_value(self):
        """
        返回当前持仓总金额
        :return: float
        """
        if self.cur_holding.__len__() == 0:
            return 0
        else:
            return sum([element.amount for _, element in self.cur_holding.items()])


class Account(object):
    """
    模拟交易回测的底层模块——账户模块
    在回测中，需要注册（register）账户后才可以进行模拟交易操作
    首先，该类需要接受environment类来完成数据的调用
    其次，该类包含了订单簿以及持仓簿以来记录账户的交易行为
    同时，该类还定义在回测中的可以使用的下单或执行交易的方法
    主要包括
    1 属性
    1.1 当前账户总金额
    1.2 当前账户现金量
    1.3 账户总金额记录簿
    1.4 交易费率假设（单向）
    1.5 订单簿
    1.6 持仓簿
    1.7 调仓簿
    2 方法
    2.1 按权重下单
    2.2 按金额下单
    2.3 按股数（份额）下单
    2.4 在处理订单前更新持仓
    2.5 在处理订单后更新持仓
    2.6 处理开（挂）订单
    """

    def __init__(self, init_invest, env, fee_rate=0.):
        """
        初始化模块
        :param init_invest: float
            初始投资额
        :param env: subclass of AbstractEnvironment
            数据环境模块，该参数需要为indcycle.backtesting.interface下
            AbstractEnvironment的子类生成的实例，用于数据的调用
        :param fee_rate: float, default 0.
            单向交易费率假设，默认0.，即无交易费率

        """
        self._cur_account_value = init_invest
        self._cash_avail = init_invest
        self._account_value_log = {}
        # assert isinstance(env, AbstractEnvironment)
        self._env = env
        self._fee_rate = fee_rate
        self._order_book = OrderBook()
        self._holding_book = HoldingBook()
        self._turnover_log = {}

    @property
    def cur_holding(self):
        """
        获取当前持仓簿中的持仓总市值
        :return: float
        """
        return self._holding_book.get_cur_holding()

    def order_by_weight(self, tick, ticker, weight, to_target=False):
        """
        按账户总市值的响应比率下单
        order_by_weight会计算需要在对应下单对象上的下单金额，并使用
        order_by_amount最终执行下单操作
        :param tick: str, datetime datetime or pandas Timestamp
            下单时间，可以转换为时间戳的时间标识
        :param ticker: str or object
            下单对象，对象字符串标识或其他对象
        :param weight: float
            下单比例
        :param to_target: bool, Default False
            是否将持仓比例调整至下单比率，默认是
            如果是且当前持仓元素中有该持仓元素，按差值进行下单
        :return: None
        """
        tick = to_datetime(tick)
        if to_target and self._holding_book.get_amount(ticker) is not None:
            open_amounts = 0.
            for _id, element in self._order_book.specific_open_orders(ticker):
                open_amounts += element.open_amount
            order_amount = weight*self._cur_account_value - self._holding_book.get_amount(ticker) - open_amounts
        else:
            order_amount = weight * self._cur_account_value
        self.order_by_amount(tick, ticker, order_amount)

    def order_by_amount(self, tick, ticker, amount, to_target=False):
        """
        按金额进行下单操作
        :param tick:  str, datetime datetime or pandas Timestamp
            下单时间，可以转换为时间戳的时间标识
        :param ticker: str or object
            下单对象，对象字符串标识或其他对象
        :param amount: float
            下单金额
        :param to_target: bool, Default False
            是否将持仓金额调整至下单金额，默认是
            如果是且当前持仓元素中有该持仓元素，按差值进行下单
        :return: None
        """
        tick = to_datetime(tick)
        if to_target and self._holding_book.get_amount(ticker) is not None:
            open_amounts = 0.
            for _id, element in self._order_book.specific_open_orders(ticker):
                open_amounts += element.open_amount
            order_amount = amount - self._holding_book.get_amount(ticker) - open_amounts
        else:
            order_amount = amount
        price = self._env.get_bar(tick, ticker, 1)
        self._order_book.open_order(ticker, tick, price, open_amount=order_amount)

    def order_by_share(self, tick, ticker, share, to_target=False):
        """
        按股数（份额）进行下单操作
        :param tick:  str, datetime datetime or pandas Timestamp
            下单时间，可以转换为时间戳的时间标识
        :param ticker: str or object
            下单对象，对象字符串标识或其他对象
        :param share: float
            下单股数（份额）
        :param to_target: bool, Default False
            是否将持股数（份额）额调整至下单股数（份额），默认是
            如果是且当前持仓元素中有该持仓元素，按差值进行下单
        :return: None
        """
        tick = to_datetime(tick)
        if to_target and self._holding_book.get_share(ticker) is not None:
            open_shares = 0.
            for _id, element in self._order_book.specific_open_orders(ticker):
                open_shares += element.open_share
            order_share = share - self._holding_book.get_share(ticker) - open_shares
        else:
            order_share = share
        price = self._env.get_bar(tick, ticker, 1)
        self._order_book.open_order(ticker, tick, price, open_share=order_share)

    def pre_action_update(self, tick):
        """
        在处理订单前更新持仓
        根据新的tick来更新上一个tick持仓信息
        :param tick: str or object
            下单对象，对象字符串标识或其他对象
        :return: None
        """
        tick = to_datetime(tick)
        self._update_holding(tick)
        self._cur_account_value = self._holding_book.total_value + self._cash_avail
        self._account_value_log.update({tick: self._cur_account_value})

    def post_action_update(self, tick):
        """
        在处理订单后更新持仓
        需要先处理开（挂）的订单
        :param tick: str or object
            下单对象，对象字符串标识或其他对象
        :return: None
        """
        tick = to_datetime(tick)
        # self._update_holding(tick)
        self._process_open_orders(tick)
        self._update_holding(tick)
        self._cur_account_value = self._holding_book.total_value + self._cash_avail
        self._account_value_log.update({tick: self._cur_account_value})

    def _update_holding(self, tick):
        """
        loop持仓簿中的每个持仓元素，根据当前tick在_env中获取每个持仓元素的最新价格并更新
        :param tick: str or object
            下单对象，对象字符串标识或其他对象
        :return: None
        """
        tick = to_datetime(tick)
        for ticker, element in self._holding_book.cur_holding.items():
            assert isinstance(element, HoldingElement)
            element.update(self._env.get_bar(tick, ticker, 1))

    def _process_open_orders(self, tick):
        """
        处理开（挂）订单
        先处理卖单，在处理买单，保证现金流通畅
        :param tick: str or object
            下单对象，对象字符串标识或其他对象
        :return: None
        """
        tick = to_datetime(tick)
        turn_over_rate = 0.
        for _id, order in self._order_book.sell_open_orders():
            assert isinstance(order, OrderElement)
            order.deal_order(tick,
                             self._env.get_bar(tick, order.ticker, 1),
                             order.open_share,
                             order.open_amount)
            self._cash_avail -= order.deal_amount
            self._holding_book.update_holding(order.ticker,
                                              order.deal_price,
                                              order.deal_share,
                                              order.deal_amount)
            turn_over_rate += abs(order.deal_amount)/self._cur_account_value

        for _id, order in self._order_book.buy_open_orders():
            assert isinstance(order, OrderElement)
            order.deal_order(tick,
                             self._env.get_bar(tick, order.ticker, 1),
                             order.open_share,
                             order.open_amount)
            self._cash_avail -= order.deal_amount
            self._holding_book.update_holding(order.ticker,
                                              order.deal_price,
                                              order.deal_share,
                                              order.deal_amount)
            turn_over_rate += abs(order.deal_amount) / self._cur_account_value
        self._turnover_log.update({tick: turn_over_rate})

    @property
    def account_log(self):
        """
        账户总市值记录
        :return: pandas.Series
        """
        return Series(self._account_value_log)

    @property
    def turnover_log(self):
        """
        账户换仓记录
        :return: pandas.Series
        """
        return Series(self._turnover_log)
