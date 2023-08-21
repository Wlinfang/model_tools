import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.style as psl

# 样式复制 ggplot
psl.use('ggplot')
# 刻度方向配置
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'inout'
# 字体配置--支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'YouYuan']
# 显示负号
plt.rcParams['axes.unicode_minus'] = False  # 负数

plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 300  # 保存的图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style
# padding bbox
plt.rcParams['figure.autolayout'] = True


def plot_line(df: pd.DataFrame, x: str, y: str, title=''):
    """
    x:y 分布图
    x: column of df  df[df[x].notna()]
    y: column of df  df[df[y].notna()]
    """
    if len(df) == 0:
        raise 'df is empty'
    df = df[(df[x].notna()) & (df[y].notna())]
    fig = plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.plot(df[x], df[y], linewidth=1, alpha=0.8, linestyle='-', marker='.')
    return fig


def plot_line_by_classier(df: pd.DataFrame, x: str, y: str, classes: list, title=''):
    """
    x:y 分布图
    classes：list,按类划分
    """
    df = df[(df[x].notna()) & (df[y].notna())]
    if classes:
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        gps = df.groupby(classes)
        lables = []
        for gp, value in gps:
            # gp: 类别 tuple ()
            # value: 某类别的值，pd.dataframe
            # color,marker,linestyle
            ax.plot(value[x], value[y], linestyle='-', marker='.', label='-'.join(gp))
            lables.append('-'.join(gp))
        # 设置 刻度范围
        ax.set_xlim(right=df[x].max() + 4)
        ax.set_ylim(top=df[y].max() + 1)
        # verticalalignment 对齐方式
        ax.set_title(title, fontsize=12, verticalalignment="baseline")
        # 上右边框不显示
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # labelspacing:条目之间的距离
        # handlelength：图例句柄的长度
        # shadow：是否为图例边框添加阴影
        ax.legend(loc='upper right', labelspacing=1, handlelength=2, fontsize=6, shadow=True)
    else:
        raise ('classes must be a list or classes not empty')
    return fig


def plot_line_and_hist(df, x, y_line, y_hist, classes=[], title=''):
    """
    同一个x,两个y，左右y轴 分布图
    y_line:线图
    y_hist：条形图
    classes：分类
    """
    fig, ax_m = plt.subplots(figsize=[12, 6])  # 设定图片大小
    if not classes:
        # classes 为空的情况
        ax_m.plot(df[x], df[y_line], marker='o', linestyle='-', label=y_line)
    else:
        gps = df.groupby(classes)
        for gp, value in gps:
            ax_m.plot(value[x], value[y_line], marker='o', linestyle='-', label='-'.join(gp))
    # 线图的 图例
    ax_m.legend(loc='upper left', labelspacing=2, handlelength=3, fontsize=8, shadow=True)
    # if x is number
    ax_m.set_xlim(right=df[x].max() + 4, left=df[x].min() - 4)
    ax_m.set_ylim(top=df[y_line].max() + 4)
    ax_m.set_ylabel(y_line)
    ax_m.set_xlabel(x)
    # plt.setp(ax1.get_xticklabels(), rotation=90, horizontalalignment='right')
    # 右轴y
    ax_r = ax_m.twinx()
    if not classes:
        # 虚化
        ax_r.bar(df[x], df[y_hist], label=y_hist, alpha=0.6)
    else:
        gps = df.groupby(classes)
        for gp, value in gps:
            ax_r.bar(value[x], value[y_hist], alpha=0.6)
    ax_r.set_ylabel(y_hist)
    plt.title(title)
    return fig


def plot_corr(df, feature_name_list):
    """
    热力图：皮尔逊系数计算
    """
    plt.figure(figsize=(8, 6))
    corr = np.corrcoef(df[feature_name_list], rowvar=False)
    # 以corr的形状生成一个全为0的矩阵
    mask = np.zeros_like(corr)
    # 将mask的对角线及以上设置为True
    # 这部分就是对应要被遮掉的部分
    mask[np.triu_indices_from(mask, k=1)] = True
    with sn.axes_style("white"):
        sn.heatmap(corr, mask=mask, fmt='.2f', annot=True, cmap="RdBu_r")
    plt.title('person corr')
    plt.show()


def rgb_to_hex(rgb):
    """
    RGB格式颜色转换为16进制颜色格式
    rgb 格式: '100, 100,100'  or (100, 100,100)
    16进制格式:'#0F0F0F'
    """
    if isinstance(rgb, str):
        RGB = rgb.split(',')  # 将RGB格式划分开来
    elif isinstance(rgb, tuple):
        RGB = list(rgb)
    elif isinstance(rgb, list):
        RGB = rgb
    else:
        raise ValueError('rgb format error ')
    if len(rgb) != 3:
        raise ValueError('rgb must having 3 value')
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


def hex_to_rgb(hex: str) -> tuple:
    '''
    16进制颜色格式颜色转换为RGB格式
    RGB 格式: '100, 100,100'  or (100, 100,100)
    hex:16进制格式:'#0F0F0F'
    返回：(100, 100,100) 两种格式的颜色
    '''
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    return tuple([r, g, b])


def gradient_color(color_list, color_sum=700, is_show=True):
    '''
    生成渐变色
    color_list：list，16进制的颜色list
    color_sum：颜色个数
    '''
    # 主颜色个数
    color_center_count = len(color_list)
    if color_sum < color_center_count * 3:
        color_sum = color_center_count * 3
    # 两个颜色之中的个数
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = hex_to_rgb(color_list[color_index_start])[1]
        color_rgb_end = hex_to_rgb(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # 生成中间渐变色
        now_color = color_rgb_start
        color_map.append(rgb_to_hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(rgb_to_hex(now_color))
        color_index_start = color_index_end

    if is_show:
        sn.palplot(color_map)
    return color_map


def gradient_color_by_palette(color_sum=10, is_show=True):
    '''
    利用调色板自定义颜色变化
    '''
    colors = sn.color_palette("Blues", n_colors=color_sum)
    # colors 中返回的rgb 在 0-1之间
    df = pd.DataFrame(np.array(colors) * 256, columns=['R', 'G', 'B'])
    df['hex'] = df.apply(lambda x: rgb_to_hex([x.R, x.G, x.B]), axis=1)
    if is_show:
        sn.palplot(df.hex)
    return df


def draw_table(ax, df: pd.DataFrame, cols: list):
    '''
    画表格图像
    ax:matplotlib.axes
    '''
    ax.set_axis_off()
    table = ax.table(df[cols].values, colLabels=cols, loc='center')
    table.auto_set_column_width(True)
    table.auto_set_font_size(True)
    # fig.savefit(bbox_inches='tight')
    # key, cell in table.get_celld().items()
    return table
