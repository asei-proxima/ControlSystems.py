from dataclasses import dataclass
import numpy as np
from numpy import float64
import numpy.typing as npt
from abc import abstractmethod, ABC, ABCMeta
from numpy.typing import NDArray
from ControlSystems.constants import G

class ControlSystem(ABC):
    """制御対象となるシステムを表すクラス"""

    # @abstractmethod
    # def state(self, t : float64) -> NDArray[float64]:
    #     """観測・介入しようとしている状態の定義。
    #     このメソッドは時刻 t を受け取ってその時点での状態をすべて返す。"""
    #     pass

    @property
    @abstractmethod
    def constants(self) -> dict[str, float64]:
        """システムの定数の値を格納する辞書"""
        pass

    @property
    @abstractmethod
    def state_dict(self) -> dict[str, int]:
        """システムの状態が、`x : NDArray` においてどのインデックスに対応するかを格納する辞書"""
        pass

    @abstractmethod
    def ssmodel(self, t : float64, x : NDArray[float64], u : float64) -> NDArray[float64]:
        """状態空間モデル。時刻`t`とその時点での状態`x`と入力`u`を受け取って、状態`x`の微分`x'(t)`を返す。
        時不変なシステムなら引数`t`は使用しないはず。

        ここでは１入力系だけを考えているので`u`はスカラー。
        """
        pass

@dataclass
class Simulator(ABC):
    """制御対象のシミュレーションを行うクラス"""

    system : ControlSystem
    """制御対象となるシステム"""

    # control_period : float64
    # """制御周期[s]"""

    initial_state : NDArray[float64]
    """初期状態"""

    control_time_series : NDArray[float64]
    """シミュレーションで使用する時刻の配列。各値の間隔は一定であることが期待される。"""

    @property
    def control_period(self) -> float64:
        """制御周期[s]"""
        return self.control_time_series[1] - self.control_time_series[0]

    def next_step_euler(self, n : int, u : float64, x : NDArray) -> NDArray[float64]:
        """オイラー法によって「次の状態」を計算する。

        Parameters:
        * n: ステップ数。ゼロから始まる。
        * u: nステップ目における入力。
        * x: nステップ目における状態。
        """
        t : float64 = self.control_time_series[n]
        x_next = x + self.system.ssmodel(t, x, u) * self.control_period
        return x_next

class VerticalDrivingArm(ControlSystem):
    """垂直駆動アーム。「Pythonによる制御工学入門」の3.1.2参照。"""

    def __init__(self, J = 0.3, M = 1.5, len = 0.7, μ = 0.1):
        self.J = J
        self.M = M
        self.l = len
        self.μ = μ

    @property
    def constants(self):
        return {
            "inertia_moment": self.J, # アームの回転軸の周りの慣性モーメント[kg⬝m²]
            "mass": self.M, # アームの質量[kg]
            "viscous_friction": self.l, # 粘性摩擦係数[Ns/m]
            "length": self.l, # アームの重心までの長さ[m]
        }

    @property
    def state_dict(self):
        return {
            "angle" : 0, # アームの角度（垂直下向きから時計回り）[rad]
            "angular_velocity" : 1  # 角速度 [rad/s]
        }

    def ssmodel(self, _t : float64, x : NDArray[float64], u : float64) -> NDArray[float64]:
        J = self.constants["inertia_moment"]
        M = self.constants["mass"]
        l = self.constants["length"]  # noqa: E741
        μ = self.constants["viscous_friction"]

        θ = self.state_dict["angle"]
        dθ = self.state_dict["angular_velocity"]

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

    def ssmodel(self, _t : float64, x : NDArray[float64], u : float64) -> NDArray[float64]:
        C1 = self.constants["cross_sectional_area_1"]
        C2 = self.constants["cross_sectional_area_2"]
        R1 = self.constants["outlet_resistance_1"]
        R2 = self.constants["outlet_resistance_2"]

        h1 = self.state_dict["water_level_1"]
        h2 = self.state_dict["water_level_2"]

        new_h1 = -(h1 / (R1 * C1)) + u / C1
        new_h2 = (h1 / (R1 * C2)) - (h2 / (R2 * C2))
        return np.array([new_h1, new_h2])

# class CartPole(ControlSystem):
#     """倒立振子"""

#     @property
#     def state_list(self):
#         return ["position", "velocity", "angle", "angular_velocity"]

#     def ssmodel(self, t : float64, x : NDArray[float64], u : float64) -> NDArray[float64]:


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