import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.axes_grid1 import host_subplot

# 刻度方向配置
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'inout'
# 字体配置--支持中文
plt.rcParams['font.sans-serif']=['SimHei','YouYuan']
# 显示负号
plt.rcParams['axes.unicode_minus']=False # 负数
# padding bbox
plt.rcParams['figure.autolayout']=True

from mylog import logutil
logger=logutil.MyLogging().get_logger(__file__)


