import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from numpy import float64
from abc import abstractmethod, ABC, ABCMeta
from numpy.typing import NDArray
from ControlSystems.constants import G
from ControlSystems.typing import Time, State

def constant_param(func):
    """制御システムの定数パラメータを表すデコレータ"""
    func.is_constant = True  # 識別用の属性を追加
    return property(func)

class ControlSystem(ABC):
    """制御対象となるシステムを表すクラス"""

    # @abstractmethod
    # def state(self, t : Time) -> NDArray[float64]:
    #     """観測・介入しようとしている状態の定義。
    #     このメソッドは時刻 t を受け取ってその時点での状態をすべて返す。"""
    #     pass

    @property
    @abstractmethod
    def constant_names(self) -> list[str]:
        """システムの定数の名前のリスト"""
        pass

    @property
    def constants(self) -> dict[str, float64]:
        """システムの定数の値の辞書"""
        return {name: getattr(self, name) for name in self.constant_names}

    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """システムの状態の名前のリスト"""
        pass

    @property
    def state_number(self) -> int:
        """状態の数"""
        return len(self.state_names)

    @abstractmethod
    def ssmodel(self, t : Time, x : State, u : float64) -> State:
        """状態空間モデル。時刻`t`とその時点での状態`x`と入力`u`を受け取って、状態`x`の微分`x'(t)`を返す。
        時不変なシステムなら引数`t`は使用しないはず。

        * ここでは１入力系だけを考えているので`u`はスカラー。
        * この時点で線形化してはいけなくて、ここでは正確なモデルを使用してください。
        """
        pass

class SystemController(ABC):
    """状態から制御入力を決定する方法を表すクラス"""

    system : ControlSystem
    """制御対象となるシステム"""

    @abstractmethod
    def control(self, t : Time, x : State) -> float64:
        """制御入力を計算する。"""
        pass


class SystemSimulator():
    """制御対象のシミュレーションを行うクラス"""

    def __init__(self, controller: SystemController, initial_state : State, time_series : NDArray[Time]):
        self.controller = controller
        self.system = controller.system

        assert len(initial_state) == len(self.system.state_names), "初期状態の次元がシステムの要求するものと異なります。"
        assert isinstance(initial_state, np.ndarray), "initial_state は numpy.ndarray 型でなければなりません"
        self.initial_state = initial_state

        self.time_series = time_series

    system : ControlSystem
    """制御対象となるシステム"""

    controller : SystemController
    """制御器"""

    initial_state : State
    """初期状態"""

    time_series : NDArray[Time]
    """シミュレーションで使用する時刻の配列。各値の間隔は一定であることが期待される。"""

    @property
    def time_interval(self) -> Time:
        """シミュレーション周期[s]"""
        return self.time_series[1] - self.time_series[0]

    def euler(self, n : int, u : float64, x : State) -> State:
        """オイラー法によって「次の状態」を計算する。

        Parameters:
        * n: ステップ数。ゼロから始まる。
        * u: nステップ目における入力。
        * x: nステップ目における状態。
        """
        t = self.time_series[n]
        x_next = x + self.system.ssmodel(t, x, u) * self.time_interval
        return x_next

    def run(self) -> NDArray[float64]:
        """シミュレーションを実行する。

        Parameters:
        * u_series: 各ステップでの入力を格納した配列。最後の要素は無視される。
        """
        controller = self.controller

        x_series = np.zeros((len(self.time_series), len(self.initial_state)))
        x_series[0] = self.initial_state

        for n, t in enumerate(self.time_series):
            if n == len(self.time_series) - 1:
                break
            u = controller.control(t, x_series[n])
            x_series[n + 1] = self.euler(n, u, x_series[n])

        return x_series


# class CartPole(ControlSystem):
#     """倒立振子"""

#     @property
#     def state_list(self):
#         return ["position", "velocity", "angle", "angular_velocity"]

#     def ssmodel(self, t : Time, x : NDArray[float64], u : float64) -> NDArray[float64]:


# @dataclass
# class ControlSystem:
#     """制御対象となるシステム"""

#     state : np.ndarray[float]
#     """観測・介入しようとしている状態。
#     float の1配列であることが期待される。
#     """

#     state_dict : dict[str, float]
#     """状態の名前と、その `ControlSystem.state` におけるインデックスの対応。"""
