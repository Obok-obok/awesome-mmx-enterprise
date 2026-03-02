from __future__ import annotations

"""Matplotlib 기본 설정(한글 폰트/기호) 유틸.

Streamlit에서 matplotlib로 한글 범례/축 라벨이 깨지는 문제를 방지합니다.
"""

from typing import Optional

import matplotlib
from matplotlib import rcParams
from matplotlib import font_manager


def _pick_font(candidates: list[str]) -> Optional[str]:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None


def configure_matplotlib_korean() -> None:
    """가능한 경우 한글 폰트를 설정합니다.

    - 서버/VM에 설치된 폰트에 따라 자동 선택
    - 없으면 기본 폰트(DejaVu)로 유지
    """
    font = _pick_font(
        [
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "NanumGothic",
            "AppleGothic",
            "Malgun Gothic",
        ]
    )
    if font:
        rcParams["font.family"] = font
    rcParams["axes.unicode_minus"] = False
