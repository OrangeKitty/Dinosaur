from pandas import read_csv, to_datetime, DataFrame

from dinosaur.backtesting.interface import AbstractEnvironment


class LocalDataEnv(AbstractEnvironment):
    """
    以本地数据为基础的environment
    数据形式需为

         tdate   000001.OF  000011.OF    ...      960028.OF  960033.OF
    2006-01-04        NaN        NaN    ...            NaN        NaN
    2006-01-05   0.016407   0.014691    ...            NaN        NaN
    2006-01-06   0.012107   0.006757    ...            NaN        NaN
    2006-01-09   0.003013   0.010547    ...            NaN        NaN
    2006-01-10   0.003975   0.010436    ...            NaN        NaN
    ...
    即第一列为唯一时间戳，后每一列为每一个ticker的收益率或净值序列，长度需和
    时间戳长度相同，且第一行为该ticker的标识

    该方法会对该形式的数据进行处理，将第一列作为pandas DataFrame的index，第一
    行作为columns并命名为fund_code；
    将时间戳全部转化为pandas Timestamp，并用时间戳的方法识别每个tick是否为周、
    月、季或年末；
    并实现接口父类的相应接口方法
    """

    def __init__(self, data_path, is_nav=True):
        """
        初始化模块
        :param data_path: str
            数据文件地址，数据文件需为csv格式
        :param is_nav: bool, default True
            数据是否为净值
        """
        self._data_lib = read_csv(data_path, parse_dates=['tdate']).set_index('tdate')
        if not is_nav:
            self._data_lib = (self._data_lib + 1).cumprod()
        self._trd_calendar = self._fetch_trade_calendar()
        # print('Environment initialized.')

    def _fetch_trade_calendar(self):
        """
        利用pandas.Timestamp的方法及属性判断每个tick的周、月、季及年的标识，并排序
        如果该标识与下个标识不同，则证明该tick为在该environment的时间轴上相应时间窗
        口的最后一天
        :return: pandas DataFrame
            输入模式如下

                        is_week_end  is_month_end  is_quarter_end  is_year_end
            tdate
            2000-01-26            0             0               0            0
            2000-01-27            0             0               0            0
            2000-01-28            1             1               0            0
            2000-02-14            0             0               0            0
            2000-02-15            0             0               0            0
        """
        calendar_frame = DataFrame(columns=['week', 'month', 'quarter', 'year'], index=self._data_lib.index)
        calendar_frame[['week', 'month', 'quarter', 'year']] = \
            [(x.week, x.month, x.quarter, x.year) for x in calendar_frame.index]
        calendar_frame[['is_week_end', 'is_month_end', 'is_quarter_end', 'is_year_end']] = \
            calendar_frame[['week', 'month', 'quarter', 'year']].diff().shift(-1).fillna(1) != 0

        return calendar_frame[['is_week_end', 'is_month_end', 'is_quarter_end', 'is_year_end']].astype(int)

    def get_bar(self, tick, ticker, counts=1):
        return float(self._data_lib.loc[:to_datetime(tick), str(ticker)].iloc[-1*counts:].values)

    def get_time_axis(self):
        return sorted(self._data_lib.index)

    def get_range_time_axis(self, start=None, end=None):
        start = min(self.get_time_axis()) if start is None else start
        end = min(self.get_time_axis()) if end is None else end
        return sorted(filter(lambda x: (x >= to_datetime(start)) & (x <= to_datetime(end)), self.get_time_axis()))

    def get_avail_ticker(self, tick):
        return list(self._data_lib.loc[to_datetime(tick), :].dropna().index)

    def is_week_end(self, tick):
        return self._trd_calendar.loc[to_datetime(tick), 'is_week_end'] == 1

    def is_month_end(self, tick):
        return self._trd_calendar.loc[to_datetime(tick), 'is_month_end'] == 1

    def is_quarter_end(self, tick):
        return self._trd_calendar.loc[to_datetime(tick), 'is_quarter_end'] == 1

    def is_year_end(self, tick):
        return self._trd_calendar.loc[to_datetime(tick), 'is_year_end'] == 1


if __name__ == '__main__':
    test_env = LocalDataEnv("D:\\Programming Platform\\Projects\\FundIndustryCycling\\data\\industrial_index.csv")
