import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from numpy import float64
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Literal

from .control_system import ControlSystem, State, Input, Time
from .system_controller import SystemController

@dataclass
class SimulationResult:
    """シミュレーションの結果"""

    system: ControlSystem
    """制御対象となるシステム"""

    time_series: NDArray[float64]
    """シミュレーションで使用した時刻の配列。各値の間隔は一定であることが期待される。"""

    result: NDArray[float64]
    """シミュレーションの結果。各行が各時刻における状態を表す。"""

    def get_data(self, name: str) -> NDArray:
        """シミュレーション結果から、指定した名前の状態データを取得する"""
        index = self.system.state_names.index(name)
        return self.result[:, index]

    def phase(self, x: str, y: str) -> tuple[Figure, Axes]:
        """シミュレーション結果から、状態を2つ選んで、その時系列発展をプロットする"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        x_series = self.get_data(x)
        y_series = self.get_data(y)
        time_series = self.time_series

        scatter = ax.scatter(x=x_series, y=y_series, cmap="inferno", c=time_series)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title("Phase plot of the system")

        # カラーバーを追加
        cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
        cbar.set_label("Time")  # カラーバーのラベルを設定
        return fig, ax


class SystemSimulator:
    """制御対象のシミュレーションを行うクラス"""

    def __init__(
        self,
        controller: SystemController,
        initial_state: State,
        time_series: NDArray,
    ):
        self.controller = controller
        self.system = controller.system

        assert len(initial_state) == len(self.system.state_names), (
            "初期状態の次元がシステムの要求するものと異なります。"
        )
        assert isinstance(initial_state, np.ndarray), (
            "initial_state は numpy.ndarray 型でなければなりません"
        )
        self.initial_state = initial_state

        self.time_series = time_series

    system: ControlSystem
    """制御対象となるシステム"""

    controller: SystemController
    """制御器"""

    initial_state: State
    """初期状態"""

    time_series: NDArray[Time]
    """シミュレーションで使用する時刻の配列。各値の間隔は一定であることが期待される。"""

    @property
    def time_interval(self) -> Time:
        """シミュレーション周期[s]"""
        return self.time_series[1] - self.time_series[0]

    def euler(self, n: int, u: Input, x: State) -> State:
        """オイラー法によって「次の状態」を計算する。

        ### Parameters:
        * n: ステップ数。ゼロから始まる。
        * u: nステップ目における入力。
        * x: nステップ目における状態。

        ### Returns:
        n+1ステップ目における状態。
        """
        t = self.time_series[n]
        x_next = x + self.system.ssmodel(t, x, u) * self.time_interval
        return x_next

    def rk4(self, n: int, u: Input, x: State) -> State:
        """古典的ルンゲ・クッタ法によって「次の状態」を計算する。"""
        t = self.time_series[n]
        Δt = self.time_interval
        r1 = self.system.ssmodel(t, x, u)
        r2 = self.system.ssmodel(t, x + (Δt / 2) * r1, u)
        r3 = self.system.ssmodel(t, x + (Δt / 2) * r2, u)
        r4 = self.system.ssmodel(t, x + Δt * r3, u)
        x_next = x + (Δt / 6) * (r1 + 2 * r2 + 2 * r3 + r4)
        return x_next

    def run(self, method : Literal['euler', 'rk4'] = 'rk4') -> SimulationResult:
        """シミュレーションを実行する。

        Parameters:
        * u_series: 各ステップでの入力を格納した配列。最後の要素は無視される。
        * method: 使用する数値積分法。'euler' または 'rk4' を指定する。デフォルトは 'rk4'。
        """
        controller = self.controller

        x_series = np.zeros((len(self.time_series), len(self.initial_state)))
        x_series[0] = self.initial_state

        next_step = (self.euler if method == 'euler' else self.rk4)

        for n, t in enumerate(self.time_series):
            if n == len(self.time_series) - 1:
                break
            u = controller.control(t, x_series[n])
            x_series[n + 1] = next_step(n, u, x_series[n])

        result = SimulationResult(self.system, self.time_series, x_series)
        return result
