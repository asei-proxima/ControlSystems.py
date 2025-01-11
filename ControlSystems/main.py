import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from numpy import float64
from abc import abstractmethod, ABC
from numpy.typing import NDArray
from ControlSystems.typing import Time, State

def constant_param(func):
    """制御システムの定数パラメータを表すデコレータ"""
    func.is_constant = True  # 識別用の属性を追加
    return property(func)

class ControlSystem(ABC):
    """制御対象となるシステムを表すクラス"""

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


@dataclass
class SimulationResult():
    """シミュレーションの結果"""

    system : ControlSystem
    """制御対象となるシステム"""

    time_series : NDArray[Time]
    """シミュレーションで使用した時刻の配列。各値の間隔は一定であることが期待される。"""

    result : NDArray[float64]
    """シミュレーションの結果。各行が各時刻における状態を表す。"""

    def phase(self, x : str, y : str) -> tuple[plt.Figure, plt.Axes]:
        """シミュレーション結果から、状態を2つ選んで、その時系列発展をプロットする"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        x_index = self.system.state_names.index(x)
        y_index = self.system.state_names.index(y)

        x_series = self.result[:, x_index]
        y_series = self.result[:, y_index]
        time_series = self.time_series

        ax.scatter(x=x_series, y=y_series, cmap="inferno", c=time_series)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title("Phase plot of the system")
        return fig, ax


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

    def run(self) -> SimulationResult:
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

        result = SimulationResult(self.system, self.time_series, x_series)
        return result
