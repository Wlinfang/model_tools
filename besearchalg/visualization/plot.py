import pandas as pd
import numpy as np


def bipart_catplot(
    tpart: np.ndarray,
    fpart: np.ndarray,
    y: np.ndarray,
    height: float = 6.27,
    aspect: float = 1.86
):
    """Short summary.

    Parameters
    ----------
    tpart : np.ndarray
        1-dim numpy vector of str.
    fpart : np.ndarray
        1-dim numpy vector of str.
    y : np.ndarray
        1-dim numpy vector of float.
    height : float
        Description of parameter `height`.
    aspect : float
        Description of parameter `aspect`.

    Returns
    -------
    type
        Description of returned object.

    """
    tmp1 = pd.DataFrame({'y': y, 'fpart': fpart, 'tpart': tpart})
    tmp1 = tmp1.sort_values(['tpart', 'fpart'])
    tmp1['cnt'] = tmp1.groupby(['tpart', 'fpart']).transform('count')
    import seaborn as sns
    g = sns.catplot(x="fpart", y="y",
                    hue="tpart",
                    data=tmp1, kind="bar", height=height, aspect=aspect)
    g.set_xticklabels(rotation=50)
    g = sns.catplot(x="fpart", y="cnt",
                    hue="tpart",
                    data=tmp1, kind="bar", height=height, aspect=aspect)
    g.set_xticklabels(rotation=50)


def ab_bipart_catplot(
    tpart: np.ndarray,
    fpart: np.ndarray,
    ab: np.ndarray,
    y: np.ndarray,
    height: float = 6.27,
    aspect: float = 1.86
):
    """Short summary.

    Parameters
    ----------
    tpart : np.ndarray
        1-dim numpy vector of str.
    fpart : np.ndarray
        1-dim numpy vector of str.
    ab : np.ndarray
        1-dim numpy vector of str.
    y : np.ndarray
        1-dim numpy vector of float.
    height : float
        Description of parameter `height`.
    aspect : float
        Description of parameter `aspect`.

    Returns
    -------
    type
        Description of returned object.

    """
    tmp1 = pd.DataFrame({'y': y, 'fpart': fpart, 'tpart': tpart, 'ab': ab})
    tmp1 = tmp1.sort_values(['tpart', 'fpart', 'ab'])
    tmp1['cnt'] = tmp1.groupby(['tpart', 'fpart', 'ab']).transform('count')
    import seaborn as sns
    g = sns.catplot(x="fpart", y="y",
                    hue="ab", col="tpart",
                    data=tmp1, kind="bar", height=height, aspect=aspect)
    g.set_xticklabels(rotation=50)
    g = sns.catplot(x="fpart", y="cnt",
                    hue="ab", col="tpart",
                    data=tmp1, kind="bar", height=height, aspect=aspect)
    g.set_xticklabels(rotation=50)
