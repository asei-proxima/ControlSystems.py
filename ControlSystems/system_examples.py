import numpy as np

from .main import ControlSystem, G, Time, State, Input

class VerticalDrivingArm(ControlSystem):
    """垂直駆動アーム。「Pythonによる制御工学入門」の3.1.2参照。"""

    def __init__(self, J : float = 0.3, M : float = 1.5, l : float = 0.7, μ : float = 0.1) -> None:  # noqa: E741
        self.J = J
        self.M = M
        self.l = l
        self.μ = μ

    J : float
    """アームの回転軸の周りの慣性モーメント[kg⬝m²]"""

    M : float
    """アームの質量[kg]"""

    l : float  # noqa: E741
    """アームの重心までの長さ[m]"""

    μ : float
    """粘性摩擦係数[Ns/m]"""

    @property
    def constant_names(self) -> list[str]:
        return ["J", "M", "l", "μ"]

    @property
    def state_names(self) -> list[str]:
        """状態リスト
        * θ: アームの角度[rad]
        * ω: アームの角速度[rad/s]
        """
        return ["θ", "ω"]

    @property
    def input_names(self) -> list[str]:
        """入力リスト
        * T: アームに与えるトルク[Nm]
        """
        return ["T"]

    def ssmodel(self, t : Time, x : State, u : Input) -> State:
        J = self.J
        M = self.M
        l = self.l  # noqa: E741
        μ = self.μ

        θ = self.get_state(x, "θ")
        ω = self.get_state(x, "ω")

        T = self.get_input(u, "T")

        dθ = ω
        dω = (- μ * ω - M * G * l * np.sin(θ) + T) / J
        return np.array([dθ, dω])


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