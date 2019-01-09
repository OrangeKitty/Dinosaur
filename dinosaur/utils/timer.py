from time import time

__all__ = ['ProgressBar', 'func_timer']


def _time_display(time_lapsed):
    """
    将时间转换为格式化字符串输出
    :param time_lapsed: float
        时间（秒）
    :return: str
        如下形式输出
        0h:1m:45s
    """
    h = time_lapsed // 3600
    time_lapsed -= 3600*h
    m = time_lapsed // 60
    time_lapsed -= 60 * m
    s = time_lapsed // 1
    return '{:d}h:{:d}m:{:d}s'.format(int(h), int(m), int(s))


class ProgressBar(object):
    """
    循环执行某loop时的进度条，在屏幕输出进度，如下

    Test
    59.32% |>>>>>>>>>>>>>>>      | 0h:0m:3s (Info)

    """

    def __init__(self, iter_items, init_phrase=None, width=100):
        """
        模块初始化
        :param iter_items: iter and has __len__ method
            iterative item且具有__len__方法
        :param init_phrase: str or None
            循环开始前的标注
        :param width: int, default 100
            进度条长度，默认100个字符输出长度
        """
        self._length = len(iter_items)
        self._counter = 0
        self._width = width
        self._start = time()

        print('{}'.format(init_phrase if init_phrase is not None else ''))

    def show(self, maker=None, proceed=True):
        """
        显示进度条
        :param maker: str or None
            当前进度标识，在进度条末端显示的当前进度主要标识
        :param proceed: bool True
            是否
        :return: None
            打印进度条
        """
        if proceed:
            self._counter += 1
        bar_context = '\r{:.2f}% |{}| {} {}'.format(
            self._counter / self._length * 100,
            '>' * self._finished_bar_counts + ' ' * (self._width - self._finished_bar_counts),
            self._timer,
            '({})'.format(maker) if maker is not None else ''
        )
        print(bar_context, end='')
        if self._counter == self._length:
            print()

    @property
    def _finished_bar_counts(self):
        """
        已完成的循环次数
        :return: int
        """
        return int((self._counter/self._length*self._width)//1)

    @property
    def _timer(self):
        """
        自循环开始已经历的时间
        :return: str
        """
        return _time_display(time() - self._start)


def func_timer(func):
    """
    function运行时间修饰器，使用如下
    @func_timer
    def function(*arg, **kwargs):
        ...
        return result

    :param func: function
    :return: function
    """

    def decorator(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print("'{} takes {}s'".format(type(func).__name__, time() - start))
        return result
    return decorator


if __name__ == '__main__':
    Iter = range(10000)
    bar = ProgressBar(Iter, 'Test')
    for i in Iter:
        bar.show(str(i), False)
        bar.show(str(i), True)
