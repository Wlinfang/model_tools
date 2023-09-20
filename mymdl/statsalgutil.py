import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def logit_fit(df_train, feature_cols, target):
    """
    statsmodels 实现 logit
    :param df_train:
    :param feature_cols:
    :param target:
    :return:
    """
    df = df_train[feature_cols + [target]]
    formula = '{} ~ {}'.format(target, ' + '.join(feature_cols))
    alg = smf.logit(formula, data=df).fit()
    print(alg.summary())
    # Odds Ratios
    df_odds_ratios = pd.DataFrame(
        {
            "OR": alg.params,
            "Lower CI": alg.conf_int()[0],
            "Upper CI": alg.conf_int()[1],
        }
    )
    df_odds_ratios = np.exp(df_odds_ratios)
    print(df_odds_ratios)
    return alg, df_odds_ratios
