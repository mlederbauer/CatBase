"""Streamlit app."""

from importlib.metadata import version

import streamlit as st

st.title(f"catbase v{version('catbase')}")  # type: ignore[no-untyped-call]
