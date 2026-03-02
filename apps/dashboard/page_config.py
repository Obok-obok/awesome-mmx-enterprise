from __future__ import annotations

"""Streamlit page configuration.

Streamlit requires # st.set_page_config (disabled)() to be called:
- exactly once per page script, and
- as the first Streamlit command in that script.

To avoid accidental double-calls (common in multipage apps),
we centralize configuration here and call it explicitly from each page entry.
"""

import streamlit as st


def configure_page(title: str) -> None:
    """페이지 설정을 구성합니다.

    Streamlit 제한사항: # st.set_page_config (disabled)()는 앱 전체에서 1회만, 엔트리포인트(app.py)에서만 호출해야 합니다.
    멀티페이지(page 스크립트)에서 호출하면 StreamlitAPIException이 발생할 수 있어 여기서는 no-op 처리합니다.
    """
    return

