import numpy as np
import pandas as pd
import seaborn as sn
import base64
import os

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
# padding bbox
plt.rcParams['figure.autolayout']=True

def plot_hist_and_line(df,x,y_hist,y_line,title='',is_show=True):
	'''
	双y轴数据
	y_hist: hist ：柱形图数据
	y_line: line : 线性图数据
	'''
	print('plot')
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
	RGB 格式: '100, 100,100'  or (100, 100,100)
	16进制格式:'#0F0F0F'
	'''
	if isinstance(rgb,str):
		RGB=rgb.split(',')  # 将RGB格式划分开来
	elif isinstance(rgb,tuple):
		RGB=list(rgb)
	elif isinstance(rgb,list):
		RGB=rgb
	else:
		raise ValueError('rgb format error ')
	if len(rgb) !=3:
		raise ValueError('rgb must having 3 value')
	color = '#'
	for i in RGB:
		num = int(i)
		# 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
		color += str(hex(num))[-2:].replace('x', '0').upper()
	return color
def Hex_to_RGB(hex):
	'''
	16进制颜色格式颜色转换为RGB格式
	RGB 格式: '100, 100,100'  or (100, 100,100)
	16进制格式:'#0F0F0F'
	返回：'100, 100,100' and (100, 100,100) 两种格式的颜色
	'''
	r = int(hex[1:3], 16)
	g = int(hex[3:5], 16)
	b = int(hex[5:7], 16)
	rgb = str(r) + ',' + str(g) + ',' + str(b)
	return rgb, [r, g, b]


def gradient_color(color_list, color_sum=700,is_show=True):
	'''
	生成渐变色
	color_list：list，16进制的颜色list
	color_sum：颜色个数
	'''
	# 主颜色个数
	color_center_count = len(color_list)
	if color_sum < color_center_count * 3:
		color_sum=color_center_count * 3
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
		color_map.append(RGB_to_Hex(now_color))
		for color_index in range(1, color_sub_count):
			now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
			color_map.append(RGB_to_Hex(now_color))
		color_index_start = color_index_end

	if is_show:
		sn.palplot(color_map)
	return color_map


def gradient_color_by_palette(color_sum=10,is_show=True):
	'''
	利用调色板自定义颜色变化
	'''
	colors=sn.color_palette("Blues",n_colors=10)
	# colors 中返回的rgb 在 0-1之间
	df=pd.DataFrame(np.array(colors)*256,columns=['R','G','B'])
	df['hex']=df.apply(lambda x:RGB_to_Hex([x.R,x.G,x.B]),axis=1)
	if is_show:
		sn.palplot(df.hex)
	return df


def draw_table(ax,df,cols):
	'''
	画表格图像
	'''
	ax.set_axis_off()
	table=ax.table(df[cols].values,colLabels=cols,loc='center')
	table.auto_set_column_width(True)
	table.auto_set_font_size(True)
	# fig.savefit(bbox_inches='tight')
	# key, cell in table.get_celld().items()
	return table

def img_to_base64(file_path,file_name):
	'''
	图片base64 编码
	如果是发送邮件，则进行图片编码'<img src="data:image/png;base64,{img_64}">'.format(img_64=img_64)
	'''
	if os.path.exists(os.path.join(file_path,file_name)):
		with open(os.path.join(file_path,file_name),'rb') as fp:
			img_data = fp.read()
			img_64 = base64.b64encode(img_data).decode()
			return img_64
