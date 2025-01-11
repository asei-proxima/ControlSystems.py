from dataclasses import dataclass
import numpy as np
from numpy import float64
import numpy.typing as npt
from abc import abstractmethod, ABC, ABCMeta
from numpy.typing import NDArray
from ControlSystems.constants import G
from ControlSystems.typing import Time, State

class ControlSystem(ABC):
    """制御対象となるシステムを表すクラス"""

    # @abstractmethod
    # def state(self, t : Time) -> NDArray[float64]:
    #     """観測・介入しようとしている状態の定義。
    #     このメソッドは時刻 t を受け取ってその時点での状態をすべて返す。"""
    #     pass

    @property
    # @abstractmethod このメソッドを実装することを矯正してもあまり子クラスにとっては嬉しくなさそう
    def constants(self) -> dict[str, float64]:
        """システムの定数の値を格納する辞書"""
        pass

    @property
    @abstractmethod
    def state_dict(self) -> dict[str, int]:
        """システムの状態が、`x : NDArray` においてどのインデックスに対応するかを格納する辞書"""
        pass

    @property
    def state_number(self) -> int:
        """状態の数"""
        return len(self.state_dict)

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
    def __init__(self, system : ControlSystem):
        self.system = system

    system : ControlSystem
    """制御対象となるシステム"""

    @abstractmethod
    def control(self, t : Time, x : State) -> float64:
        """制御入力を計算する。"""
        pass

class SystemSimulator():
    """制御対象のシミュレーションを行うクラス"""

    def __init__(self, system : ControlSystem, initial_state : State, time_series : NDArray[float64]):
        self.system = system

        assert len(initial_state) == len(system.state_dict), "初期状態の次元がシステムの要求するものと異なります。"
        assert isinstance(initial_state, np.ndarray), "initial_state は numpy.ndarray 型でなければなりません"
        self.initial_state = initial_state

        self.time_series = time_series

    system : ControlSystem
    """制御対象となるシステム"""

    initial_state : State
    """初期状態"""

    time_series : NDArray[float64]
    """シミュレーションで使用する時刻の配列。各値の間隔は一定であることが期待される。"""

    @property
    def time_interval(self) -> float64:
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

    def run(self, controller : SystemController) -> NDArray[float64]:
        """シミュレーションを実行する。

        Parameters:
        * u_series: 各ステップでの入力を格納した配列。最後の要素は無視される。
        """
        assert self.system == controller.system, "制御対象のシステムが制御器とシミュレータで一致しません。"

        x_series = np.zeros((len(self.time_series), len(self.initial_state)))
        x_series[0] = self.initial_state

        for n, t in enumerate(self.time_series):
            if n == len(self.time_series) - 1:
                break
            u = controller.control(t, x_series[n])
            x_series[n + 1] = self.euler(n, u, x_series[n])

        return x_series

class VerticalDrivingArm(ControlSystem):
    """垂直駆動アーム。「Pythonによる制御工学入門」の3.1.2参照。"""

    def __init__(self, J = 0.3, M = 1.5, len = 0.7, μ = 0.1):
        self.J = J
        self.M = M
        self.l = len
        self.μ = μ

    J : float64
    """アームの回転軸の周りの慣性モーメント[kg⬝m²]"""

    M : float64
    """アームの質量[kg]"""

    l : float64
    """アームの重心までの長さ[m]"""

    μ : float64
    """粘性摩擦係数[Ns/m]"""

    # θ : float64
    # """アームの角度（垂直下向きから時計回り）[rad]"""

    # dθ : float64
    # """アームの角速度 [rad/s]"""

    @property
    def constants(self):
        return {
            "J": self.J,
            "M": self.M,
            "l": self.l,
            "μ": self.μ,
        }

    @property
    def state_dict(self):
        return {
            "θ" : 0, # アームの角度（垂直下向きから時計回り）[rad]
            "dθ" : 1  # 角速度 [rad/s]
        }

    def ssmodel(self, _t : Time, x : NDArray[float64], u : float64) -> NDArray[float64]:
        J = self.J
        M = self.M
        l = self.l  # noqa: E741
        μ = self.μ

        θ = self.state_dict["θ"]
        dθ = self.state_dict["dθ"]

        new_θ = dθ
        new_dθ = (- μ * dθ - M * G * l * np.sin(θ) + u) / J
        return np.array([new_θ, new_dθ])


class TwoTanks(ControlSystem):
    """直列に結合した2つのタンク。「はじめての現代制御理論」の1.5節参照。
    ただし、ここで示すモデルは既に線形化されている。
    """

    def __init__(self, C1 = 1.0, C2 = 2.0, R1 = 0.1, R2 = 0.1):
        self.C1 = C1
        self.C2 = C2
        self.R1 = R1
        self.R2 = R2

    @property
    def constants(self):
        return {
            "cross_sectional_area_1": self.C1, # タンク1の断面積[m²]
            "cross_sectional_area_2": self.C2, # タンク2の断面積[m²]
            "outlet_resistance_1": self.R1, # タンク1の出口抵抗 [s/m²]
            "outlet_resistance_2": self.R2, # タンク2の出口抵抗 [s/m²]
        }

    @property
    def state_dict(self):
        return {
            "water_level_1": 0, # タンク1の水位[m]
            "water_level_2": 1  # タンク2の水位[m]
        }

    def ssmodel(self, _t : Time, x : NDArray[float64], u : float64) -> NDArray[float64]:
        C1 = self.constants["cross_sectional_area_1"]
        C2 = self.constants["cross_sectional_area_2"]
        R1 = self.constants["outlet_resistance_1"]
        R2 = self.constants["outlet_resistance_2"]

        h1 = self.state_dict["water_level_1"]
        h2 = self.state_dict["water_level_2"]

        dh1 = -(h1 / (R1 * C1)) + u / C1
        dh2 = (h1 / (R1 * C2)) - (h2 / (R2 * C2))
        return np.array([dh1, dh2])

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

if __name__ == "__main__":
    arm = VerticalDrivingArm()
    print(arm.constants)
    print(arm.state_dict)