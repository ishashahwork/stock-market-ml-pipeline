"""Streamlit entry point: a runnable demo with separate real-data pages."""

import streamlit as st

def main():
    st.set_page_config(page_title="Stock Market ML · Dashboard", page_icon="📈", layout="wide")
    loading = st.empty()
    loading.info("Loading the dashboard. The first visit after inactivity can take a minute.")
    from src.dashboard import render_dashboard

    loading.empty()
    render_dashboard()


if __name__ == "__main__":
    main()
