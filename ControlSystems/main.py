from dataclasses import dataclass
import numpy as np
from numpy import float64
import numpy.typing as npt
from abc import abstractmethod, ABC
from numpy.typing import NDArray
from ControlSystems.constants import G

class ControlSystem(ABC):
    """制御対象となるシステムを表す抽象クラス"""

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
        """状態空間モデル。時刻`t`とその時点での状態`x`と入力`u`を受け取って、状態`x`の微分`x'(t)`を返す。"""
        pass


class VerticalDrivingArm(ControlSystem):
    """垂直駆動アーム。「Pythonによる制御工学入門」の3.1.2参照。"""

    @property
    def constants(self):
        return {
            "inertia_moment": 2.0, # アームの回転軸の周りの慣性モーメント[kg⬝m^2]
            "mass": 1.0, # アームの質量[kg]
            "viscous_friction": 0.1, # 粘性摩擦係数[Ns/m]
            "length": 1.0, # アームの重心までの長さ[m]
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
