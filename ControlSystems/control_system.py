import numpy as np
from numpy.typing import NDArray
from typing import Literal, TypeAlias
from abc import abstractmethod, ABC

State : TypeAlias = NDArray
"""システムのある時刻での状態を全部集めたベクトル"""

Input : TypeAlias = NDArray
"""ある時刻での制御入力を集めたベクトル"""

Time : TypeAlias = float
"""時刻や時間を表す型。単位は秒[s]"""

class ControlSystem(ABC):
    """制御対象となるシステムを表すクラス"""

    @property
    @abstractmethod
    def constant_names(self) -> list[str]:
        """システムの定数の名前のリスト"""
        pass

    @property
    def constants(self) -> dict[str, float]:
        """システムの定数の値の辞書"""
        return {name: getattr(self, name) for name in self.constant_names}

    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """システムの状態の名前のリスト"""
        pass

    def get_state(self, x : State, name: str) -> float:
        """与えられた名前の状態の値を取得する"""
        index = self.state_names.index(name)
        return x[index]

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """システムの入力の名前のリスト"""
        pass

    def get_input(self, u : Input, name: str) -> float:
        """与えられた名前の入力の値を取得する"""
        index = self.input_names.index(name)
        return u[index]

    @abstractmethod
    def ssmodel(self, t: Time, x: State, u: Input) -> State:
        """状態空間モデル。時刻`t`とその時点での状態`x`と入力`u`を受け取って、状態`x`の微分`x'(t)`を返す。
        * 多入力多出力系を考えている。
        * ここでは一般的なシステムを考えているが、時不変なシステムなら引数`t`は使用しないはず。
        * この時点では線形化しないように。ここでは正確なモデルを使用するべき。
        """
        pass