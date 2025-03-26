import numpy as np

from .system_controller import SystemController
from .control_system import ControlSystem, State, Input, Time
from numpy.typing import NDArray

G = 9.80665
"""重力加速度 [m/s^2]"""

class TrivialController(SystemController):
    """常にゼロ入力を返すような自明なコントローラ"""

    def __init__(self, system : SystemController) -> None:
        self.system = system

    def contol(self, t : Time, x : State) -> Input:
        u_shape = len(self.system.input_names)
        return np.zeros(u_shape)


class SingleInputPController(SystemController):
    """比例制御を行うコントローラ

    制御入力の次元が1であるような系にだけ適用される。
    """

    def __init__(self, system : ControlSystem, ref : State, Kp : NDArray) -> None:
        assert len(system.input_names) == 1, "制御入力の次元が1である必要があります。"
        self.system = system
        self.ref = ref
        self.Kp = Kp

    Kp : NDArray
    """比例ゲイン。状態 x と同じ次元数である必要がある。"""

    def contol(self, t : Time, x : State) -> Input:
        assert len(self.kp) == len(x), "状態と比例ゲインの次元が一致しません。"

        # 目標値からのずれを計算
        error = self.ref - x

        u = np.dot(self.Kp, error)
        return np.array([u])