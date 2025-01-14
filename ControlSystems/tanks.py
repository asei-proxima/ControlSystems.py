import numpy as np

from controlsystems.typedef import Time, State, Input
from controlsystems.main import ControlSystem
from numpy import float64
from numpy.typing import NDArray

class TwoTanks(ControlSystem):
    """直列に結合した2つのタンク。「はじめての現代制御理論」の1.5節参照。
    ただし、ここで示すモデルは既に線形化されている。
    """

    def __init__(self, C1 : float, C2 : float, R1 : float, R2 : float) -> None:
        self.C1 = C1
        self.C2 = C2
        self.R1 = R1
        self.R2 = R2

    C1 : float
    """タンク1の断面積[m²]"""

    C2 : float
    """タンク2の断面積[m²]"""

    R1 : float
    """タンク1の出口抵抗 [s/m²]"""

    R2 : float
    """タンク2の出口抵抗 [s/m²]"""

    @property
    def constant_names(self) -> list[str]:
        return ["C1", "C2", "R1", "R2"]

    @property
    def state_names(self) -> list[str]:
        return ["h1", "h2"]

    def ssmodel(self, t : Time, x : State, u : Input) -> State:
        C1 = self.C1
        C2 = self.C2
        R1 = self.R1
        R2 = self.R2

        h1 = self.get_state(x, "h1")
        h2 = self.get_state(x, "h2")

        dh1 = -(h1 / (R1 * C1)) + u / C1
        dh2 = (h1 / (R1 * C2)) - (h2 / (R2 * C2))
        return np.array([dh1, dh2])
