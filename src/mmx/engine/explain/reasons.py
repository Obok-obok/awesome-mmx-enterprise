from __future__ import annotations
from typing import List

def reasons_for_channel(channel: str, sat: float, mroi: float, half_life_days: float) -> List[str]:
    out: List[str] = []
    if sat >= 0.85:
        out.append(f"{channel}: 포화도 {sat:.2f}로 포화 구간(추가 집행 효율 낮음)")
    else:
        out.append(f"{channel}: 포화도 {sat:.2f}로 성장 여지(추가 집행 효율 여지)")
    out.append(f"{channel}: 한계 ROI(mROI) {mroi:.6f} (현재 수준에서의 추가 효율)")
    out.append(f"{channel}: Adstock 반감기 {half_life_days:.1f}일 (효과 지속 기간)")
    return out
