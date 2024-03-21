"""Streamlit app."""

from importlib.metadata import version

import streamlit as st

st.title(f"cat_base v{version('cat-base')}")  # type: ignore[no-untyped-call]
