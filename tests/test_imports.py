def test_core_modules_importable():
    # If any Streamlit page imports are missing, this test fails early.
    import mmx.formatting  # noqa: F401
    import mmx.tracking  # noqa: F401
    # Streamlit-dependent modules are only import-checked when streamlit is available.
    try:
        import streamlit as _st  # noqa: F401
    except Exception:
        return
    import mmx.optimizer  # noqa: F401
    import mmx.executive  # noqa: F401


def test_formatting_handles_none_and_nan():
    import math

    from mmx.formatting import format_int, format_ratio, format_won

    assert format_won(None) == "-"
    assert format_int(None) == "-"
    assert format_ratio(None) == "-"
    assert format_ratio(float("nan")) == "-"
    assert format_ratio(float("inf")) == "-"
    assert format_ratio(1.2345, ndigits=2) == "1.23"
    assert format_won(23000000) == "23,000,000원"
    assert format_int(23000000) == "23,000,000"
    assert math.isfinite(float(format_ratio(1.0)))
