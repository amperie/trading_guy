"""
Data provider that jsut reads the SPXU data file locally
"""
import pandas as pd

from data_providers.data_provider import DataProvider

class TestDataProvider(DataProvider):

    def __init__(self, cfg: dict=None):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        fp = self.cfg['path']
        tr = self.cfg['truncate']
        if tr>0:
            df = pd.read_csv(fp, nrows=tr)
        else:
            df = pd.read_csv(fp)
        self.data = df