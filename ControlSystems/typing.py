
from numpy.typing import NDArray
from typing import TypeAlias
from numpy import float64
import numpy as np

State : TypeAlias = NDArray
"""システムのある時刻での状態を全部集めたベクトル"""

Input : TypeAlias = NDArray
"""ある時刻での制御入力を集めたベクトル"""

Time : TypeAlias = float64
"""時刻や時間を表す型。単位は秒[s]"""
