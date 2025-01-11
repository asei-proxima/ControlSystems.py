import numpy as np

from ControlSystems.constants import G
from ControlSystems.typing import Time, State
from ControlSystems.main import ControlSystem
from numpy import float64

class VerticalDrivingArm(ControlSystem):
    """垂直駆動アーム。「Pythonによる制御工学入門」の3.1.2参照。"""

    def __init__(self, J = 0.3, M = 1.5, l = 0.7, μ = 0.1):  # noqa: E741
        self.J = J
        self.M = M
        self.l = l
        self.μ = μ

    J : float64
    """アームの回転軸の周りの慣性モーメント[kg⬝m²]"""

    M : float64
    """アームの質量[kg]"""

    l : float64  # noqa: E741
    """アームの重心までの長さ[m]"""

    μ : float64
    """粘性摩擦係数[Ns/m]"""

    # θ : float64
    # """アームの角度（垂直下向きから時計回り）[rad]"""

    # dθ : float64
    # """アームの角速度 [rad/s]"""

    @property
    def constant_names(self) -> list[str]:
        return ["J", "M", "l", "μ"]

    @property
    def state_names(self):
        return ["θ", "dθ"]

    def ssmodel(self, _t : Time, x : State, u : float64) -> State:
        J = self.J
        M = self.M
        l = self.l  # noqa: E741
        μ = self.μ

        θ_index = self.state_names.index("θ")
        dθ_index = self.state_names.index("dθ")
        θ = x[θ_index]
        dθ = x[dθ_index]

        new_θ = dθ
        new_dθ = (- μ * dθ - M * G * l * np.sin(θ) + u) / J
        return np.array([new_θ, new_dθ])
