import unittest
import sys
import os

import numpy as np

sys.path.append('../../')
from utils import plotutil
import pandas as pd

class MyTestCase(unittest.TestCase):
    def test_plot_line(self):
        x = np.arange(0,100)
        y = 3 * x
        z = list(zip(x, y))
        df = pd.DataFrame(z, columns=['x', 'y'])
        plotutil.plot_line(df, 'x', 'y', 'test')

if __name__ == '__main__':
    unittest.main()