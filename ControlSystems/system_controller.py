import numpy as np
from numpy.typing import NDArray
from typing import Literal, TypeAlias
from abc import abstractmethod, ABC

from .control_system import ControlSystem, State, Input, Time

class SystemController(ABC):
    """状態から制御入力を決定する方法を表すクラス"""

    system: ControlSystem
    """制御対象となるシステム"""

    ref : State
    """制御の目標となる状態の値"""

    @abstractmethod
    def control(self, t: Time, x: State) -> Input:
        """制御入力を計算する。"""
        pass