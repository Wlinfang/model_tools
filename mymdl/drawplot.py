from pyplotz.pyplotz import PyplotZ
from pyplotz.pyplotz import plt
from data.analyis import datacal
import seaborn as sns
import pandas as pd

plt.rc('figure',figsize=(8,6))
font_options={
    'weight':'bold',
    'size':'14'
}
plt.rc('font',**font_options)


def liftchart(df,x,y,classes='',bin=10,title='',xlabel='',ylabel=''):
    '''
    x:x轴；y:y轴
    :param df:dataframe
    :param x:
    :param y:
    :param classes:分组，str
    :param bin:
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    '''

    # #== 单个TODO 待输出
    plt.close('all')
    if classes !='':
        df_out = datacal.cal_accume(df, x, y, bin, classes=[classes])
        #== 显示样本数量
        df_fig = pd.pivot_table(df_out, index=classes, columns=['lbl', 'grid'],
                                 values=['count'], aggfunc=['mean'])
        df_fig=df_fig['mean']['count']
        #== 行数
        rows=df_fig.index.tolist()
        n_rows=len(rows)
        # 列数
        cols=df_fig.columns.levels[0].categories.to_tuples().tolist()
        n_cols=len(cols)
        cell_text=df_fig.values.tolist()
        plt.subplot(2, 1,1)
        draw_lineplot(df_out,'grid','mean',hue=classes,title=title,xlabel=xlabel,ylabel=ylabel)
        plt.subplot(2, 1, 2)
        draw_lineplot(df_out,'grid','acmMean',hue=classes,title=title+'累计',xlabel=xlabel,ylabel=ylabel)
    else :
        df_out = datacal.cal_accume(df, x, y, bin)
        plt.subplot(2, 1, 1)
        draw_lineplot(df_out, 'grid','mean', title=title, xlabel=xlabel, ylabel=ylabel)
        plt.subplot(2, 1, 2)
        draw_lineplot(df_out, 'grid','acmMean', title=title+'累计', xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    # plt.show()
    return plt



def univarchart(df,x,y,bin=10,classes='',title='',xlabel='',ylabel=''):
    '''
    特征与label的关系图,y为label
    :param df:
    :return:
    '''
    plt.close('all')
    plt.subplot(1, 1, 1)
    if classes !='':
        df_out = datacal.cal_univar(df, x, y, bin, classes=[classes])
        draw_lineplot(df_out,'grid','mean',hue=classes,title=title,xlabel=xlabel,ylabel=ylabel)
    else:
        df_out = datacal.cal_univar(df, x, y, bin)
        draw_lineplot(df_out, 'grid', 'mean', title=title, xlabel=xlabel, ylabel=ylabel)
    # plt.show()
    return plt

def pdpchart(df,x,y,bin=10,classes='',title='',xlabel='模型分',ylabel='逾期率'):
    '''
    特征与label的关系图,y为label
    :param df:
    :return:
    '''
    plt.close('all')
    plt.subplot(1, 1, 1)

    if classes !='':
        df_out = datacal.cal_univar(df, x, y, bin, classes=[classes])
        draw_lineplot(df_out,'grid','mean',hue=classes,title=title,xlabel=xlabel,ylabel=ylabel)
    else:
        df_out = datacal.cal_univar(df, x, y, bin)
        draw_lineplot(df_out, 'grid', 'mean', title=title, xlabel=xlabel, ylabel=ylabel)
    # plt.show()
    return plt


def draw_barplot(df,x,y,hue='',title=''):
    '''
    :param df: dataframe
    :param x: 横坐标
    :param y: 纵坐标
    :param hue: 分类
    :param title:
    :return:fig
    '''
    pltz = PyplotZ()
    pltz.enable_chinese()
    fig = plt.figure()
    plt.close('all')
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    if hue != '':
        sns.barplot(x, y, hue=hue, data=df, ax=ax)
    else:
        sns.barplot(x, y, data=df, ax=ax)
    # pltz.xticks(range(len(df[x].unique().tolist())), df[x].unique().tolist())
    pltz.xlabel(x)
    pltz.ylabel(y)
    pltz.title(title)
    pltz.legend()
    plt.grid()
    # plt.show()
    return fig


def draw_lineplot(df,x,y,hue='',title='',xlabel='',ylabel=''):
    '''
    :param df: dataframe
    :param x: 横坐标
    :param y: 纵坐标
    :param hue: 分类
    :param title:
    :return:fig
    '''
    pltz = PyplotZ()
    pltz.enable_chinese()
    # fig = plt.figure()
    if hue != '':
        for type in df[hue].unique().tolist():
            # == 画图
            tmp=df[df[hue]==type]
            plt.plot(tmp[x], tmp[y], linestyle='dashed', marker='o',label=type)
    else:
        plt.plot(df[x], df[y], linestyle='dashed', marker='o')
    # pltz.xticks(range(len(df[x].unique().tolist())), df[x].unique().tolist())
    if xlabel !='':
        pltz.xlabel(xlabel)
    else:
        pltz.xlabel(x)
    if ylabel !='':
        pltz.ylabel(ylabel)
    else:
        pltz.ylabel(y)
    pltz.title(title)
    pltz.legend()
    plt.grid()
    # plt.show()
    return plt