import pandas as pd
import seaborn as sn 
import matplotlib.pyplot as plt
import matplotlib.style as psl 
from mpl_toolkits.axes_grid1 import host_subplot
# 样式复制 ggplot
psl.use('ggplot') 
# 刻度方向配置
plt.rcParams['xtick.direction'] = 'out' 
plt.rcParams['ytick.direction'] = 'inout' 
# 字体配置--支持中文
plt.rcParams['font.sans-serif']=['SimHei','YouYuan']
# 显示负号
plt.rcParams['axes.unicode_minus']=False # 负数

plt.rcParams['figure.figsize'] = (8.0, 4.0) # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 300 # 保存的图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style


def plot_hist_and_line(df,x,y_hist,y_line,title='',is_show=True):
	'''
	双y轴数据
	y_hist: hist ：柱形图数据
	y_line: line : 线性图数据
	'''
	fig=plt.figure(figsize=(12,6))
	ax1 = fig.add_subplot(111)  
	ax1.plot(df[x], df[y_line],'or-',label=y_line)
	ax1.legend(loc='upper left',labels=[y_line])
	plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

	ax2 = ax1.twinx() # this is the important function  
	ax2.bar(df[x],df[y_hist],alpha=0.3,color='blue',label=y_hist)
	ax2.legend(loc='upper right',labels=[y_hist])
	# plt.title(title)
	plt.title(title,backgroundcolor='#3c7f99',fontsize=24, weight='bold',color='white')
	plt.xticks(rotation=45,fontsize=12,weight='bold')

	if is_show:
		plt.show()
	return plt  

def plot_line_with_doubley(df,x,y1,y2,x_label=None,y1_label=None,y2_label=None,title=''):
	'''
	同1个x轴，2个y轴，均为线
	'''
	if x_label is None:
		x_label=x
	if y1_label is None:
		y1_label=y1
	if y2_label is None:
		y2_label=y2
	fig = plt.figure(figsize=[12,6])
	ax1 = host_subplot(111)
	ax2 = ax1.twinx()
	
	ax1.set_xlabel(x)
	ax1.set_ylabel(y1_label)
	ax2.set_ylabel(y2_label)
	
	p1, = ax1.plot(df[x], df[y1], label=y1_label)
	p2, = ax2.plot(df[x], df[y2], label=y2_label)
	
	leg = plt.legend(loc='best')
	ax1.yaxis.get_label().set_color(p1.get_color())
	leg.texts[0].set_color(p1.get_color())

	ax2.yaxis.get_label().set_color(p2.get_color())
	leg.texts[1].set_color(p2.get_color())
	
	plt.title(title)

	# plt.show()
	return fig


def RGB_to_Hex(rgb):
	'''
	RGB格式颜色转换为16进制颜色格式
	RGB 格式: '100, 100,100'
	16进制格式:'#0F0F0F'
	'''
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color
def Hex_to_RGB(hex):
	'''
	16进制颜色格式颜色转换为RGB格式

	'''
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    return rgb, [r, g, b]



def gradient_color(color_list, color_sum=700):
	'''
	生成渐变色
	color_list：list，16进制的颜色list
	color_sum：颜色个数
	'''
	# 主颜色个数
    color_center_count = len(color_list)
    # 两个颜色之中的个数
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # 生成中间渐变色
        now_color = color_rgb_start
        color_map.append(RGB_list_to_Hex(now_color))
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            color_map.append(RGB_list_to_Hex(now_color))
        color_index_start = color_index_end
    return color_map