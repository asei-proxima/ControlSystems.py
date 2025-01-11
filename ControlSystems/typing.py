from numpy.typing import NDArray
from typing import TypeAlias
from numpy import float64

State : TypeAlias = NDArray[float64]
"""システムの状態"""

Time : TypeAlias = float64
"""時刻や時間を表す型。単位は秒[s]"""
