"""
Streamlit Dashboard for Monetary Policy Transmission & Financial Resilience.

Interactive UI for running rate shocks, visualizing NLP stance scores,
and analyzing household/MSME financial resilience under policy transmission.
"""

import pandas as pd
import streamlit as st

from ai.policy_analyser import PolicyAnalyser
from analysis.scenario_analysis import ScenarioEngine
from models.resilience_model import ResilienceEngine

st.set_page_config(
    page_title="Monetary Policy & Resilience Dashboard",
    page_icon="🏦",
    layout="wide",
)

# Initialize engines
@st.cache_resource
def get_engines():
    return PolicyAnalyser(), ScenarioEngine(), ResilienceEngine()

policy_analyser, scenario_engine, resilience_engine = get_engines()

# Header
st.title("🏦 Monetary Policy & Financial Resilience Engine")
st.markdown(
    "Simulate RBI monetary policy rate shocks, analyze policy stance from statement text, "
    "and measure household and MSME financial resilience."
)

st.divider()

# Sidebar - Controls
st.sidebar.header("⚙️ Scenario Controls")

repo_rate_change_bps = st.sidebar.slider(
    "Repo Rate Change (bps)",
    min_value=-200.0,
    max_value=200.0,
    value=50.0,
    step=25.0,
    help="Adjust policy rate shift in basis points (e.g., +50 bps hike or -25 bps cut).",
)

st.sidebar.subheader("Transmission Factors")
borrowing_transmission = st.sidebar.slider(
    "Lending Rate Pass-Through",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Fraction of repo rate change passed to borrowing rates.",
)

deposit_transmission = st.sidebar.slider(
    "Deposit Rate Pass-Through",
    min_value=0.0,
    max_value=1.0,
    value=0.68,
    step=0.05,
    help="Fraction of repo rate change passed to deposit rates.",
)

# Tabs Layout
tab_scenario, tab_nlp, tab_profiles = st.tabs(
    ["📊 Integrated Scenario", "🔍 NLP Stance Analyser", "👤 Economic Profiles"]
)

# ---------------- Tab 1: Integrated Scenario ----------------
with tab_scenario:
    st.header("Integrated Scenario Simulation")

    sample_text = st.text_area(
        "RBI Policy Statement / Speech Text",
        value=(
            "The Monetary Policy Committee noted that inflation risk remains elevated "
            "due to food price volatility, necessitating sustained focus on price stability. "
            "The committee reiterated its stance on withdrawal of accommodation to align inflation."
        ),
        height=120,
    )

    if st.button("Run Simulation", type="primary"):
        results = scenario_engine.run_scenario(
            statement_text=sample_text,
            repo_rate_change_bps=repo_rate_change_bps,
            borrowing_transmission_factor=borrowing_transmission,
            deposit_transmission_factor=deposit_transmission,
        )

        summary = results["scenario_summary"]
        impacts = results["profile_impacts"]

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Policy Stance Label", summary["policy_stance_label"])
        col2.metric("Stance Score", f"{summary['policy_stance_score']} / 100")
        col3.metric("Lending Rate Delta", f"{summary['borrowing_rate_shock_bps']:+} bps")
        col4.metric("Deposit Rate Delta", f"{summary['deposit_rate_shock_bps']:+} bps")

        st.divider()

        # Visualizations
        col_chart1, col_chart2 = st.columns(2)

        # Resilience Index Chart
        profile_names = [k.replace("_", " ").title() for k in impacts.keys()]
        resilience_scores = [v["resilience_index"] for v in impacts.values()]
        df_resilience = pd.DataFrame(
            {"Profile": profile_names, "Resilience Index Score": resilience_scores}
        )

        with col_chart1:
            st.subheader("Financial Resilience Index (0–100)")
            st.bar_chart(df_resilience.set_index("Profile"))

        # Monthly EMI Delta Chart
        emi_deltas = [v["monthly_emi_change"] for v in impacts.values()]
        df_emi = pd.DataFrame({"Profile": profile_names, "Monthly EMI Delta (₹)": emi_deltas})

        with col_chart2:
            st.subheader("Monthly EMI Change (₹)")
            st.bar_chart(df_emi.set_index("Profile"))

        # Detailed Breakdown Table
        st.subheader("Detailed Profile Impact Table")
        df_table = pd.DataFrame.from_dict(impacts, orient="index")
        st.dataframe(df_table, use_container_width=True)


# ---------------- Tab 2: NLP Policy Analyser ----------------
with tab_nlp:
    st.header("Standalone NLP Stance Analyser")

    text_to_analyze = st.text_area(
        "Enter RBI Statement or Speech Text for NLP Parsing",
        value="There is no inflation risk and no rate hike planned. Demand weakness requires liquidity support.",
        height=140,
    )

    negation_window = st.slider("Negation Window (words)", min_value=1, max_value=5, value=3)

    if st.button("Analyze Stance"):
        analyser = PolicyAnalyser(negation_window=negation_window)
        output = analyser.analyze_statement(text_to_analyze)

        col1, col2, col3 = st.columns(3)
        col1.metric("Classified Stance", output["stance_label"])
        col2.metric("Net Score", f"{output['net_stance_score']} / 100")
        col3.metric("Hawkish vs Dovish Weights", f"{output['weighted_hawkish_score']} : {output['weighted_dovish_score']}")

        st.subheader("Signals Extracted")
        c1, c2 = st.columns(2)
        with c1:
            st.success("**Hawkish Signals:**")
            st.write(output["hawkish_signals_found"] or "None")
        with c2:
            st.info("**Dovish Signals:**")
            st.write(output["dovish_signals_found"] or "None")


# ---------------- Tab 3: Economic Profiles ----------------
with tab_profiles:
    st.header("Built-In Indian Economic Profiles")
    profiles = resilience_engine.default_profiles

    profile_data = []
    for key, p in profiles.items():
        profile_data.append({
            "ID": key,
            "Name": p.name,
            "Category": p.category,
            "Monthly Income (₹)": f"₹{p.monthly_income:,.0f}",
            "Liquid Savings (₹)": f"₹{p.liquid_savings:,.0f}",
            "Floating Debt (₹)": f"₹{p.floating_debt_principal:,.0f}",
            "Borrowing Rate": f"{p.current_borrowing_rate * 100:.1f}%",
            "Deposit Rate": f"{p.current_deposit_rate * 100:.1f}%",
        })

    st.table(pd.DataFrame(profile_data))
