"""
Data provider that jsut reads the test data file locally
"""
import pandas as pd
from pathlib import Path

from data_providers.data_provider import DataProvider

class TestDataProvider(DataProvider):

    def __init__(self, cfg: dict=None):
        super().__init__(cfg)
        self.load_data()

    def load_data(self):
        fp = self.cfg['path']
        tr = self.cfg['truncate']

        # Convert to Path and resolve relative to project root if not absolute
        fp_path = Path(fp)
        if not fp_path.is_absolute():
            # Get project root (parent of data_providers directory)
            project_root = Path(__file__).parent.parent
            fp_path = project_root / fp

        if tr>0:
            df = pd.read_csv(fp_path, nrows=tr)
        else:
            df = pd.read_csv(fp_path)
        self.data = df