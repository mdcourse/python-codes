from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy, sys, os

import warnings
warnings.filterwarnings('ignore')

class Outputs:
    def __init__(self,
                 data_folder = "./",
                 *args,
                 **kwargs):
        self.data_folder = data_folder
        super().__init__(*args, **kwargs)

        if os.path.exists(self.data_folder) is False:
            os.mkdir(self.data_folder)
