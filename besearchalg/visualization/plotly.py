import typing
from typing import Union, Mapping
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def plotly_two_y(
    x1: Union[np.ndarray, Mapping[str, np.ndarray]],
    y1bars: Mapping[str, np.ndarray],
    x2: Union[np.ndarray, Mapping[str, np.ndarray]],
    y2lines: Mapping[str, np.ndarray],
    xlabel='xaxis title',
    y1label='y1axis title',
    y2label='y2axis title',
    title='figure title',
    figsize=(800, 600),
    renderer=None
):
    """Short summary.
    https://plotly.com/python/multiple-axes/
    https://plotly.com/python/bar-charts/

    Parameters
    ----------
    x1 : np.ndarray
        Description of parameter `x1`.
    y1bars : typing.Mapping[str, np.ndarray]
        Description of parameter `y1bars`.
    x2 : np.ndarray
        Description of parameter `x2`.
    y2lines : typing.Mapping[str, np.ndarray]
        Description of parameter `y2lines`.
    xlabel : type
        Description of parameter `xlabel`.
    y1label : type
        Description of parameter `y1label`.
    y2label : type
        Description of parameter `y2label`.
    title : type
        Description of parameter `title`.

    Returns
    -------
    type
        Description of returned object.

    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    for x, ylines, secondary_y, _func in [(x1, y1bars, False, go.Bar),
                                          (x2, y2lines, True, go.Scatter)]:
        if isinstance(x, typing.Mapping):
            x_all = np.sort(np.unique(np.hstack([_ for _ in x.values()])))
            df_all = pd.DataFrame(index=x_all)
        for k, v in ylines.items():
            if isinstance(x, typing.Mapping):
                df_all[k] = df_all.index.map(pd.Series(index=x[k], data=v))
                fig.add_trace(
                    _func(x=x_all, y=df_all[k].values, name=k),
                    secondary_y=secondary_y,
                )
            else:
                fig.add_trace(
                    _func(x=x, y=v, name=k),
                    secondary_y=secondary_y,
                )

    # Add figure title
    fig.update_layout(
        title_text=title,
        width=figsize[0],
        height=figsize[1]
    )

    # Set x-axis title
    fig.update_xaxes(title_text=xlabel)

    # Set y-axes titles
    fig.update_yaxes(title_text=y1label, secondary_y=False)
    fig.update_yaxes(title_text=y2label, secondary_y=True)
    if renderer is None:
        fig.show()
    else:
        # https://github.com/jupyter/nbconvert/issues/944
        fig.show(renderer=renderer)  # browser, notebook


def plotly_x_y(
    x: typing.Union[np.ndarray, typing.Mapping],
    ylines: typing.Mapping[str, np.ndarray],
    xlabel='xaxis title',
    ylabel='y1axis title',
    title='figure title',
    figsize=(800, 600),
    style='line',
    renderer=None
):
    if style.lower() == 'line':
        _func = go.Scatter
    elif style.lower() == 'bar':
        _func = go.Bar
    else:
        raise Exception('Unkown plotly style')
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    # Add traces
    if isinstance(x, typing.Mapping):
        x_all = np.sort(np.unique(np.hstack([_ for _ in x.values()])))
        df_all = pd.DataFrame(index=x_all)
    for k, v in ylines.items():
        if isinstance(x, typing.Mapping):
            df_all[k] = df_all.index.map(pd.Series(index=x[k], data=v))
            fig.add_trace(
                _func(x=x_all, y=df_all[k].values, name=k),
                secondary_y=False,
            )
        else:
            fig.add_trace(
                _func(x=x, y=v, name=k),
                secondary_y=False,
            )
    # Add figure title
    fig.update_layout(
        title_text=title,
        width=figsize[0],
        height=figsize[1]
    )
    # Set x-axis title
    fig.update_xaxes(title_text=xlabel)
    # Set y-axes titles
    fig.update_yaxes(title_text=ylabel, secondary_y=False)
    if renderer is None:
        fig.show()
    else:
        # https://github.com/jupyter/nbconvert/issues/944
        fig.show(renderer=renderer)  # browser, notebook
