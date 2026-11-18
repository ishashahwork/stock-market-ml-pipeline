"""Streamlit entry point: a runnable demo with separate real-data pages."""

import streamlit as st

from src.dashboard import render_dashboard


def main():
    st.set_page_config(page_title="Stock Market ML · Dashboard", page_icon="📈", layout="wide")
    render_dashboard()


if __name__ == "__main__":
    main()
