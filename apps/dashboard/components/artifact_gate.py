from __future__ import annotations

"""Artifact gate utilities for the dashboard.

The dashboard is an *operations* UI. It must never silently show empty pages.
If required artifacts are missing, it should:
- explain what's missing (Korean, non-technical),
- show the expected generation command(s),
- optionally provide a safe one-click runner for local/dev usage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    path: Path
    required: bool = True
    hint: str = ""


def render_artifact_gate(
    *,
    checks: Iterable[ArtifactCheck],
    run_buttons: Optional[dict[str, list[str]]] = None,
    header: str = "데이터/아티팩트 상태",
) -> bool:
    """Render gate UI and return whether all required artifacts exist.

    Args:
        checks: Files/directories to check.
        run_buttons: Optional mapping from button label to shell command list.
            Example: {"recommend 실행": ["python", "scripts/recommend.py"]}.
        header: UI header.

    Returns:
        True if all required artifacts exist, else False.
    """
    st.subheader(header)

    all_ok = True
    rows = []
    for c in checks:
        exists = c.path.exists()
        if c.required and not exists:
            all_ok = False
        rows.append(
            {
                "항목": c.name,
                "상태": "✅ 있음" if exists else ("⚠️ 없음" if c.required else "⏳ 없음(선택)"),
                "경로": str(c.path),
                "설명": c.hint,
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    if all_ok:
        st.success("필수 데이터/아티팩트가 준비되었습니다.")
        return True

    st.warning(
        "필수 데이터/아티팩트가 없어 일부 화면을 표시할 수 없습니다. "
        "아래 안내에 따라 파이프라인을 실행해 주세요."
    )

    if run_buttons:
        st.markdown("#### 빠른 실행(로컬/개발 환경)")
        st.caption("서버/권한 정책에 따라 버튼 실행이 제한될 수 있습니다. 제한될 경우, 명령어를 터미널에서 실행하세요.")
        for label, cmd in run_buttons.items():
            with st.expander(label, expanded=False):
                st.code(" ".join(cmd))
                if st.button(f"▶ {label}", key=f"run_{label}"):
                    _run_command(cmd)

    return False


def _run_command(cmd: list[str]) -> None:
    import subprocess
    import sys

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        st.code(proc.stdout[-4000:] if proc.stdout else "", language="text")
        if proc.stderr:
            st.code(proc.stderr[-4000:], language="text")
        if proc.returncode == 0:
            st.success("실행 완료. 페이지를 새로고침(F5) 해주세요.")
        else:
            st.error(f"실행 실패 (exit={proc.returncode}). 터미널에서 로그를 확인하세요.")
    except Exception as e:
        st.error(f"실행 중 예외가 발생했습니다: {e}")
