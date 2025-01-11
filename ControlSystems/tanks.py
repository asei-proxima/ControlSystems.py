import numpy as np

from ControlSystems.constants import G
from ControlSystems.typing import Time, State
from ControlSystems.main import ControlSystem
from numpy import float64
from numpy.typing import NDArray

class TwoTanks(ControlSystem):
    """直列に結合した2つのタンク。「はじめての現代制御理論」の1.5節参照。
    ただし、ここで示すモデルは既に線形化されている。
    """

    def __init__(self, C1 = 1.0, C2 = 2.0, R1 = 0.1, R2 = 0.1):
        self.C1 = C1
        self.C2 = C2
        self.R1 = R1
        self.R2 = R2

    C1 : float64
    """タンク1の断面積[m²]"""

    C2 : float64
    """タンク2の断面積[m²]"""

    R1 : float64
    """タンク1の出口抵抗 [s/m²]"""

    R2 : float64
    """タンク2の出口抵抗 [s/m²]"""

    @property
    def constant_names(self):
        return ["C1", "C2", "R1", "R2"]

    @property
    def state_names(self):
        return ["h1", "h2"]

    def ssmodel(self, _t : Time, x : NDArray[float64], u : float64) -> NDArray[float64]:
        C1 = self.C1
        C2 = self.C2
        R1 = self.R1
        R2 = self.R2

        h1_index = self.state_names.index("h1")
        h2_index = self.state_names.index("h2")
        h1 = x[h1_index]
        h2 = x[h2_index]

        dh1 = -(h1 / (R1 * C1)) + u / C1
        dh2 = (h1 / (R1 * C2)) - (h2 / (R2 * C2))
        return np.array([dh1, dh2])
