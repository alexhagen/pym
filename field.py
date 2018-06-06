import pandas as pd
import numpy as np
from pyg import threed as pyg3d


class field(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(field, self).__init__(*args, **kwargs)
        if 'name' not in kwargs.keys():
            kwargs['name'] = field
        self.name = kwargs['name']

    @property
    def _constructor(self):
        return field

    def add_data(self, x=None, y=None, z=None):
        if x is None:
            pass
        if y is None:
            pass
        if z is None:
            raise ValueError('You didn\'t pass any new data')
        try:
            _ = [_x for _x in x]
        except TypeError:
            x = [x]
        try:
            _ = [_y for _y in y]
        except TypeError:
            y = [y]
        z = np.array(z)

        for i, _x in enumerate(x):
            if str(_x) not in self.columns.values:
                self[str(_x)] = pd.Series(data=z[:, i], index=y)

        for j, _y in enumerate(y):
            if _y not in self.index:
                new_row_np = np.nan * np.ones_like(self.iloc[-1])
                for i, _x in enumerate(x):
                    bool_array = \
                        [float(val) == _x for val in self.columns.values]
                    idx = np.argwhere(bool_array)
                    new_row_np[idx] = z[j, i]
                self.loc[_y] = new_row_np
                self.reset_index()
        return self

    def surf(self, transpose=False):
        self._plot = pyg3d.pyg3d()
        if transpose:
            _x = [float(val) for val in self.index.values]
            _y = [float(val) for val in self.columns.values]
            _z = np.zeros_like(self.values.T)
            for i, row in enumerate(self.values.T):
                for j, col in enumerate(row):
                    _z[i, j] = float(col)
        else:
            _x = [float(val) for val in self.columns.values]
            _y = [float(val) for val in self.index.values]
            _z = np.zeros_like(self.values)
            for i, row in enumerate(self.values):
                for j, col in enumerate(row):
                    _z[i, j] = float(col)
        self._plot.surf(_x, _y, _z)
        return self._plot
