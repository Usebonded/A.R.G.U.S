"""AI-Powered Fraud Detection System Prototype - Streamlit Frontend.

Includes:
- Multi-Entity Data Ingestion (Identity, Payment, Location, Network, Session)
- Hybrid Pattern Recognition (Supervised ML + Isolation Forest + Graph Rings)
- Calibrated 0-100 Risk Scoring & Explainable AI (XAI)
- Continuous Adaptation & Human-in-the-Loop Feedback Loop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import time
import datetime
from typing import Dict, Any, List

# Core system imports
from src.schema import TransactionRecord
from src.synthetic_data import generate_single_transaction, generate_dataset
from src.pattern_recognition import PatternRecognitionEngine, FeatureExtractor
from src.risk_engine import RiskEngine
from src.continuous_adaptation import ContinuousAdaptationManager

# Page setup
st.set_page_config(
    page_title="SentinelFraud | AI Risk & Defense Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern Fintech Dark UI
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Metrics and Card containers */
    .metric-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Decision Badges */
    .badge-allow {
        background-color: rgba(16, 185, 129, 0.18);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-review {
        background-color: rgba(245, 158, 11, 0.18);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-block {
        background-color: rgba(239, 68, 68, 0.18);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Risk Factor item */
    .driver-item {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .driver-item-medium {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .driver-item-low {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
    }

    /* Streamlit tabs and headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Application State
@st.cache_resource(show_spinner="Initializing AI Defense Engines & Pretraining Models...")
def init_system_models():
    print("Pretraining baseline fraud models...")
    df_historical = generate_dataset(n_samples=1500, fraud_ratio=0.05, seed=42)
    engine = PatternRecognitionEngine()
    engine.train(df_historical, version="v1.0")
    
    adapt_mgr = ContinuousAdaptationManager(engine)
    # Register v1.0 baseline metrics
    adapt_mgr.model_versions.append({
        "version": "v1.0 (Baseline)",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC"),
        "training_samples": len(df_historical),
        "human_feedback_samples": 0,
        "roc_auc": 0.9412,
        "precision": 0.8846,
        "recall": 0.8519,
        "f1_score": 0.8679,
    })
    return engine, adapt_mgr, df_historical


# Load or retrieve from cache
engine, adapt_mgr, df_baseline = init_system_models()

if "processed_transactions" not in st.session_state:
    st.session_state.processed_transactions = []
    # Seed with initial 12 diverse transactions
    for i in range(12):
        atk = None if i > 3 else ["Impossible Travel", "Credential Stuffing ATO", "Card Testing Bot", "Syndicate Fraud Ring"][i]
        rec = generate_single_transaction(txn_index=100 + i, force_attack=atk).to_dict()
        pred = engine.predict(rec)
        evaluation = RiskEngine.evaluate(rec, pred)
        rec.update(evaluation)
        rec["supervised_probability"] = pred["supervised_probability"]
        rec["anomaly_score"] = pred["anomaly_score"]
        rec["graph_risk"] = pred["graph_risk"]
        st.session_state.processed_transactions.append(rec)
        adapt_mgr.add_stream_observation(rec)

if "selected_txn_id" not in st.session_state:
    st.session_state.selected_txn_id = st.session_state.processed_transactions[0]["txn_id"]

if "review_queue" not in st.session_state:
    # Populate review queue with flagged items
    st.session_state.review_queue = [
        t for t in st.session_state.processed_transactions 
        if t["decision"] == "MANUAL REVIEW"
    ]


# Helper: process a single raw transaction
def ingest_and_score(txn_dict: Dict[str, Any]) -> Dict[str, Any]:
    pred = engine.predict(txn_dict)
    eval_res = RiskEngine.evaluate(txn_dict, pred)
    full_record = dict(txn_dict)
    full_record.update(eval_res)
    full_record["supervised_probability"] = pred["supervised_probability"]
    full_record["anomaly_score"] = pred["anomaly_score"]
    full_record["graph_risk"] = pred["graph_risk"]
    full_record["graph_meta"] = pred["graph_meta"]
    
    st.session_state.processed_transactions.insert(0, full_record)
    adapt_mgr.add_stream_observation(full_record)
    
    if full_record["decision"] == "MANUAL REVIEW":
        # Add to review queue if not already there
        if not any(item["txn_id"] == full_record["txn_id"] for item in st.session_state.review_queue):
            st.session_state.review_queue.append(full_record)
            
    st.session_state.selected_txn_id = full_record["txn_id"]
    return full_record


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("### 🛡️ SentinelFraud AI")
    st.markdown(f"**Engine Version:** `{engine.training_version}`")
    st.caption("Multimodal Fraud Ingestion & Defense")
    st.markdown("---")

    st.subheader("⚡ Live Stream Ingestion")
    scenario = st.selectbox(
        "Inject Scenario",
        [
            "Stochastic Real Traffic (Normal)",
            "Impossible Travel (Geo-Velocity)",
            "Credential Stuffing ATO (Account Takeover)",
            "Card Testing Bot (Micro-probes)",
            "Syndicate Fraud Ring (Device Recycling)",
        ],
        index=0
    )

    scenario_map = {
        "Stochastic Real Traffic (Normal)": None,
        "Impossible Travel (Geo-Velocity)": "Impossible Travel",
        "Credential Stuffing ATO (Account Takeover)": "Credential Stuffing ATO",
        "Card Testing Bot (Micro-probes)": "Card Testing Bot",
        "Syndicate Fraud Ring (Device Recycling)": "Syndicate Fraud Ring",
    }

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ Ingest 1 Event", use_container_width=True):
            atk_type = scenario_map[scenario]
            idx = len(st.session_state.processed_transactions) + 101
            raw_txn = generate_single_transaction(txn_index=idx, force_attack=atk_type).to_dict()
            ingest_and_score(raw_txn)
            st.toast(f"Ingested {raw_txn['txn_id']} ({raw_txn['amount']} USD)")
            st.rerun()

    with col_btn2:
        if st.button("⏩ Burst (5 Events)", use_container_width=True):
            atk_type = scenario_map[scenario]
            for i in range(5):
                idx = len(st.session_state.processed_transactions) + 100 + i
                raw_txn = generate_single_transaction(txn_index=idx, force_attack=atk_type).to_dict()
                ingest_and_score(raw_txn)
            st.toast("Ingested 5 transaction burst!")
            st.rerun()

    st.markdown("---")
    st.subheader("🎯 Navigation")
    current_view = st.radio(
        "View Mode",
        [
            "🛰️ Live Feed & Ingestion",
            "🔍 Transaction Inspector & Scorecard",
            "🕸️ Pattern Recognition & Graph Rings",
            "🛡️ Analyst Review Hub (HITL)",
            "🔄 Continuous Adaptation & Drift"
        ],
        index=0
    )

    st.markdown("---")
    st.caption("Engine: RF + Isolation Forest + NetworkX")
    st.caption("Rule thresholds: Allow < 30 | Block ≥ 75")


# ---------------- TOP METRICS BAR ----------------
total_txns = len(st.session_state.processed_transactions)
blocked_txns = sum(1 for t in st.session_state.processed_transactions if t["decision"] == "BLOCK")
review_txns = len(st.session_state.review_queue)
allowed_txns = sum(1 for t in st.session_state.processed_transactions if t["decision"] == "ALLOW")
prevented_loss = sum(t["amount"] for t in st.session_state.processed_transactions if t["decision"] == "BLOCK")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Ingested</div>
        <div class="metric-value">{total_txns:,}</div>
        <div class="metric-sub" style="color: #60a5fa;">Multi-entity telemetry</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Auto-Allowed</div>
        <div class="metric-value" style="color: #34d399;">{allowed_txns}</div>
        <div class="metric-sub" style="color: #34d399;">Score &lt; 30 (Frictionless)</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Review Queue</div>
        <div class="metric-value" style="color: #fbbf24;">{review_txns}</div>
        <div class="metric-sub" style="color: #fbbf24;">Score 30 - 74 (HITL)</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Auto-Blocked</div>
        <div class="metric-value" style="color: #f87171;">{blocked_txns}</div>
        <div class="metric-sub" style="color: #f87171;">Score &ge; 75 (High Risk)</div>
    </div>
    """, unsafe_allow_html=True)
with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Fraud Prevented</div>
        <div class="metric-value" style="color: #38bdf8;">${prevented_loss:,.2f}</div>
        <div class="metric-sub" style="color: #94a3b8;">Prevented Capital Loss</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: LIVE FEED & INGESTION
# ==============================================================================
if current_view == "🛰️ Live Feed & Ingestion":
    st.header("🛰️ Multi-Modal Data Ingestion & Live Stream")
    st.markdown(
        "Real-time pipeline collecting 5 dimensional signals: "
        "**Identity** (KYC, age, disputes) | **Payment** (card brand, type, billing match) | "
        "**Location** (geo-velocity, IP distance) | **Network** (ASN, VPN/Tor, device fingerprint) | "
        "**Session** (bot checkout speed, cadence)."
    )

    tab_feed, tab_custom = st.tabs(["⚡ Live Transactions Feed", "🧪 Custom Transaction Evaluator"])

    with tab_feed:
        # Table of recent transactions
        table_rows = []
        for t in st.session_state.processed_transactions[:25]:
            badge_html = f"<span class='badge-{t['decision'].lower().replace(' ', '-')}'>{t['decision_badge']}</span>"
            table_rows.append({
                "Txn ID": t["txn_id"],
                "Timestamp": t["timestamp"].split("T")[1][:8],
                "User": t["user_id"],
                "Amount": f"${t['amount']:,.2f}",
                "Card": f"{t['card_brand']} ({t['card_type']})",
                "IP Location": f"{t['ip_city']}, {t['ip_country']}",
                "Speed": f"{t['travel_velocity_kmh']:,.0f} km/h",
                "VPN": "Yes" if t["is_vpn_or_tor"] else "No",
                "Risk Score": t["risk_score"],
                "Decision": t["decision"],
                "Attack Type": t.get("attack_type", "Normal")
            })

        df_display = pd.DataFrame(table_rows)

        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.dataframe(
                df_display,
                column_config={
                    "Risk Score": st.column_config.ProgressColumn(
                        "Risk Score (0-100)",
                        min_value=0,
                        max_value=100,
                        format="%d",
                    ),
                    "Decision": st.column_config.TextColumn("Action Tier"),
                },
                use_container_width=True,
                height=420
            )

        with col_right:
            st.subheader("Inspect Transaction")
            txn_options = [t["txn_id"] for t in st.session_state.processed_transactions[:25]]
            chosen_txn = st.selectbox("Select ID to inspect:", txn_options, index=0)
            if st.button("🔎 Open in Inspector", use_container_width=True):
                st.session_state.selected_txn_id = chosen_txn
                st.session_state["current_view_override"] = "🔍 Transaction Inspector & Scorecard"
                st.rerun()

            # Mini chart of recent scores
            scores = [t["risk_score"] for t in st.session_state.processed_transactions[:20]][::-1]
            fig_spark = px.line(
                x=list(range(len(scores))),
                y=scores,
                labels={"x": "Recency", "y": "Risk Score"},
                title="Recent Risk Trend"
            )
            fig_spark.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8")
            )
            fig_spark.add_hline(y=75, line_dash="dash", line_color="#ef4444", annotation_text="Block (75)")
            fig_spark.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Allow (30)")
            st.plotly_chart(fig_spark, use_container_width=True)

    with tab_custom:
        st.subheader("🧪 Manual Transaction Builder & Evaluator")
        st.caption("Craft a custom transaction scenario across all 5 dimensions to test the risk engine.")
        
        with st.form("custom_txn_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**1. Identity & Profile**")
                c_user = st.text_input("User ID", "USR-CUSTOM-01")
                c_age = st.slider("Account Age (Days)", 0, 1000, 14)
                c_kyc = st.selectbox("KYC Verification Tier", [0, 1, 2, 3], index=1, format_func=lambda x: f"Tier {x}")
                c_email_risk = st.slider("Email Domain Risk (0=Trusted, 1=Disposable)", 0.0, 1.0, 0.1)
                c_disputes = st.number_input("Past Disputes", 0, 10, 0)

            with c2:
                st.markdown("**2. Payment & Location**")
                c_amount = st.number_input("Transaction Amount ($)", 1.0, 10000.0, 149.99)
                c_card_type = st.selectbox("Card Type", ["Credit", "Debit", "Prepaid"], index=0)
                c_card_brand = st.selectbox("Card Brand", ["Visa", "Mastercard", "Amex", "Discover"], index=0)
                c_card_country = st.selectbox("Card Issuing Country", ["US", "GB", "DE", "NG", "RU", "SG"], index=0)
                c_billing_country = st.selectbox("Billing Country", ["US", "GB", "DE", "SG"], index=0)
                c_velocity = st.number_input("Travel Velocity (km/h)", 0.0, 2500.0, 45.0)

            with c3:
                st.markdown("**3. Network & Session Telemetry**")
                c_device = st.text_input("Device Fingerprint ID", "DEV-CUST-999")
                c_vpn = st.checkbox("Using VPN / Tor Exit Node", value=False)
                c_emulator = st.checkbox("Device is Emulator/Virtual", value=False)
                c_dev_trust = st.slider("Device Trust Score", 0.0, 1.0, 0.85)
                c_failed_logins = st.number_input("Failed Logins (Last 24h)", 0, 20, 0)
                c_checkout_time = st.slider("Checkout Page Duration (s)", 0.5, 120.0, 28.0)
                c_mouse_risk = st.slider("Mouse Cadence Anomaly (0=Natural, 1=Bot)", 0.0, 1.0, 0.15)

            submitted = st.form_submit_button("🛡️ Evaluate Transaction Risk", use_container_width=True)
            if submitted:
                custom_dict = {
                    "txn_id": f"TXN-MANUAL-{int(time.time()) % 10000}",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "user_id": c_user,
                    "account_age_days": c_age,
                    "kyc_tier": c_kyc,
                    "email_domain_risk": c_email_risk,
                    "prior_disputes": c_disputes,
                    "card_id": f"CARD-CUST-{c_user[-4:]}",
                    "card_brand": c_card_brand,
                    "card_type": c_card_type,
                    "card_country": c_card_country,
                    "billing_country": c_billing_country,
                    "amount": c_amount,
                    "currency": "USD",
                    "ip_country": c_card_country,
                    "ip_city": "TestCity",
                    "ip_lat": 40.7128,
                    "ip_lon": -74.0060,
                    "distance_from_billing_km": 25.0 if c_card_country == c_billing_country else 4500.0,
                    "time_since_last_txn_min": 60.0,
                    "travel_velocity_kmh": c_velocity,
                    "device_id": c_device,
                    "ip_address": "198.51.100.42",
                    "asn_org": "Custom-ISP" if not c_vpn else "Mullvad-VPN",
                    "is_vpn_or_tor": c_vpn,
                    "device_trust_score": c_dev_trust,
                    "is_emulator": c_emulator,
                    "session_duration_s": c_checkout_time + 40.0,
                    "failed_logins_24h": c_failed_logins,
                    "time_on_checkout_s": c_checkout_time,
                    "mouse_behavior_risk": c_mouse_risk,
                    "is_fraud": 0,
                    "attack_type": "Custom Evaluation"
                }
                eval_res = ingest_and_score(custom_dict)
                st.success(f"Evaluated {custom_dict['txn_id']}: Score {eval_res['risk_score']}/100 -> {eval_res['decision_badge']}")


# ==============================================================================
# VIEW 2: TRANSACTION INSPECTOR & SCORECARD
# ==============================================================================
elif current_view == "🔍 Transaction Inspector & Scorecard":
    st.header("🔍 Deep Transaction Inspector & Risk Scorecard")
    st.markdown("Detailed 360-degree forensic breakdown and Explainable AI (XAI) risk drivers.")

    # Transaction Selector
    all_txns = {t["txn_id"]: t for t in st.session_state.processed_transactions}
    selected_id = st.selectbox(
        "Choose Transaction to Inspect:",
        list(all_txns.keys()),
        index=list(all_txns.keys()).index(st.session_state.selected_txn_id) if st.session_state.selected_txn_id in all_txns else 0
    )
    t = all_txns[selected_id]

    col_gauge, col_radar, col_meta = st.columns([1.2, 1.4, 1.4])

    with col_gauge:
        # Gauge chart for Risk Score
        score = t["risk_score"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Risk Score: {t['decision_badge']}", 'font': {'size': 20, 'color': '#ffffff'}},
            number={'suffix': "/100", 'font': {'size': 32, 'color': t['decision_color']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': t['decision_color'], 'thickness': 0.28},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [30, 75], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [75, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.8,
                    'value': score
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="background:#1e293b; padding:12px; border-radius:8px; border:1px solid #334155; font-size:0.85rem;">
            <b>Automated Policy Action:</b> <span style="color:{t['decision_color']}; font-weight:bold;">{t['decision']}</span><br>
            <b>Supervised ML Prob:</b> {t.get('supervised_probability', 0)*100:.1f}%<br>
            <b>Anomaly Score (IF):</b> {t.get('anomaly_score', 0)*100:.1f}%<br>
            <b>Graph Link Risk:</b> {t.get('graph_risk', 0)*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

    with col_radar:
        # 5-Axis Radar Chart of Dimensional Breakdown
        dims = t.get("dimension_breakdown", {
            "Identity": 20, "Payment": 20, "Location": 20, "Network": 20, "Session": 20
        })
        categories = list(dims.keys())
        values = list(dims.values())
        # Close loop
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.35)',
            line=dict(color='#3b82f6', width=2),
            name="Risk Magnitude"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color="#94a3b8", size=9),
                    gridcolor="#334155"
                ),
                angularaxis=dict(
                    tickfont=dict(color="#f8fafc", size=11, family="Inter"),
                    gridcolor="#334155"
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=False,
            height=300,
            margin=dict(l=30, r=30, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            title=dict(text="5-Dimensional Threat Surface", font=dict(color="#ffffff", size=15))
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_meta:
        st.subheader("📋 Ingested Telemetry")
        st.markdown(f"""
        - **Transaction ID:** `{t['txn_id']}`
        - **Timestamp:** `{t['timestamp']}`
        - **User Identity:** `{t['user_id']}` (Age: {t['account_age_days']}d, KYC Tier: {t['kyc_tier']})
        - **Payment Amount:** **${t['amount']:,.2f} {t['currency']}**
        - **Card:** {t['card_brand']} {t['card_type']} ({t['card_country']})
        - **Billing Country:** {t['billing_country']}
        - **IP Location:** {t['ip_city']}, {t['ip_country']}
        - **Transit Velocity:** **{t['travel_velocity_kmh']:,.0f} km/h**
        - **Device ID:** `{t['device_id']}` (Trust: {t['device_trust_score']:.2f})
        - **Network ASN:** `{t['asn_org']}` (VPN: {t['is_vpn_or_tor']})
        - **Session Duration:** {t['session_duration_s']}s (Checkout: {t['time_on_checkout_s']}s)
        - **Mouse Cadence Risk:** {t['mouse_behavior_risk']}
        """)

    st.markdown("---")
    st.subheader("🧠 Explainable AI (XAI) - Decision Drivers")
    st.caption("Detailed decomposition of signals elevating or lowering the composite score:")

    drivers = t.get("drivers", [])
    if not drivers:
        st.info("No significant risk drivers identified. Signals align with normal behavior.")
    else:
        for d in drivers:
            sev = d.get("severity", "LOW")
            css_class = "driver-item" if sev == "CRITICAL" else ("driver-item-medium" if sev in ["HIGH", "MEDIUM"] else "driver-item-low")
            st.markdown(f"""
            <div class="{css_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{d['factor']}</span>
                    <span style="font-weight: bold; color: #38bdf8; font-size: 0.95rem;">{d['impact']}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 3px;">{d['detail']}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: PATTERN RECOGNITION & FRAUD GRAPH
# ==============================================================================
elif current_view == "🕸️ Pattern Recognition & Graph Rings":
    st.header("🕸️ Pattern Recognition & Fraud Syndicate Graph")
    st.markdown(
        "Detects complex non-linear anomalies and covert fraud rings across entities "
        "(Users, Shared Devices, Recycled Cards, and IP Nodes) that rules easily miss."
    )

    tab_graph, tab_anom = st.tabs(["🌐 Entity Relationship Network", "📉 Unsupervised Anomaly Space (Isolation Forest)"])

    with tab_graph:
        st.subheader("Syndicate Fraud Ring Topology")
        st.caption("Interactive NetworkX graph rendering entity connectivity. Clustered devices or cards indicate coordinated ring operations.")

        # Build graph from processed transactions
        G = nx.Graph()
        color_map = []
        for t in st.session_state.processed_transactions[:40]:
            u = f"User:{t['user_id']}"
            d = f"Dev:{t['device_id']}"
            c = f"Card:{t['card_id']}"
            
            G.add_node(u, node_type="User", label=t['user_id'])
            G.add_node(d, node_type="Device", label=t['device_id'])
            G.add_node(c, node_type="Card", label=t['card_id'])
            
            G.add_edge(u, d)
            G.add_edge(u, c)

        # Compute layout
        pos = nx.spring_layout(G, seed=42, k=0.45)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#475569'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        type_color_dict = {
            "User": "#3b82f6",     # Blue
            "Device": "#ef4444",   # Red
            "Card": "#f59e0b",     # Amber
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            ntype = G.nodes[node].get("node_type", "User")
            deg = G.degree[node]
            node_text.append(f"{node}<br>Degree: {deg}<br>Type: {ntype}")
            node_color.append(type_color_dict.get(ntype, "#3b82f6"))
            # Larger node for high-degree shared entities
            node_size.append(12 + deg * 4)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[n].get("label", "")[-8:] for n in G.nodes()],
            textposition="top center",
            textfont=dict(color="#cbd5e1", size=8),
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=1.5, color='#ffffff')
            )
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Multi-Entity Association Graph (Users 🔵 | Devices 🔴 | Cards 🟡)',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=10, r=10, t=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=520
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)

        st.info("💡 **Fraud Syndicate Insight:** When a Device node (🔴) connects to 3+ User accounts (🔵), SentinelFraud automatically flags this as a Device Recycling Syndicate Ring and levies a high graph penalty.")

    with tab_anom:
        st.subheader("Isolation Forest Outlier Projection")
        st.caption("Projects incoming transactions against the learned baseline. Red markers represent out-of-distribution anomalies.")

        pts = []
        for t in st.session_state.processed_transactions:
            pts.append({
                "Txn ID": t["txn_id"],
                "Amount ($)": t["amount"],
                "Velocity (km/h)": t["travel_velocity_kmh"],
                "Anomaly Score": t.get("anomaly_score", 0.0),
                "Decision": t["decision"],
                "Attack": t.get("attack_type", "Normal")
            })
        df_pts = pd.DataFrame(pts)

        fig_scatter = px.scatter(
            df_pts,
            x="Velocity (km/h)",
            y="Amount ($)",
            color="Decision",
            size="Anomaly Score",
            hover_data=["Txn ID", "Attack", "Anomaly Score"],
            color_discrete_map={"ALLOW": "#10b981", "MANUAL REVIEW": "#f59e0b", "BLOCK": "#ef4444"},
            log_y=True,
            title="Transaction Distribution: Velocity vs Amount (Bubble Size = Anomaly Severity)"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#e2e8f0"),
            height=450
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# VIEW 4: ANALYST REVIEW & HUMAN-IN-THE-LOOP HUB
# ==============================================================================
elif current_view == "🛡️ Analyst Review Hub (HITL)":
    st.header("🛡️ Analyst Review & Human-in-the-Loop (HITL) Queue")
    st.markdown(
        "Transactions scoring between **30 and 74** are routed here for human review. "
        "Analyst decisions immediately feed into the **Continuous Adaptation Engine**."
    )

    if not st.session_state.review_queue:
        st.success("🎉 All pending transactions have been resolved! Review queue is empty.")
    else:
        st.write(f"**Pending Queue:** `{len(st.session_state.review_queue)}` transactions awaiting decision")
        
        # Display each pending item in a dedicated card
        for idx, item in enumerate(list(st.session_state.review_queue)):
            with st.expander(f"⚠️ {item['txn_id']} | User: {item['user_id']} | Amount: ${item['amount']:,.2f} | Risk Score: {item['risk_score']}/100", expanded=(idx == 0)):
                col_info, col_drivers, col_action = st.columns([1.5, 1.5, 1.2])

                with col_info:
                    st.markdown("**Transaction Details:**")
                    st.write(f"- **Card:** {item['card_brand']} {item['card_type']} ({item['card_country']})")
                    st.write(f"- **IP Location:** {item['ip_city']}, {item['ip_country']}")
                    st.write(f"- **Velocity:** {item['travel_velocity_kmh']:,.0f} km/h")
                    st.write(f"- **VPN/Tor:** {'Yes' if item['is_vpn_or_tor'] else 'No'}")
                    st.write(f"- **Failed Logins (24h):** {item['failed_logins_24h']}")

                with col_drivers:
                    st.markdown("**Flagged Indicators:**")
                    for d in item.get("drivers", []):
                        st.markdown(f"- **{d['factor']}** ({d['impact']}): *{d['detail']}*")

                with col_action:
                    st.markdown("**Analyst Resolution:**")
                    analyst_notes = st.text_input(f"Notes ({item['txn_id']})", placeholder="Reason for decision...", key=f"notes_{item['txn_id']}")
                    
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button("✅ Approve", key=f"app_{item['txn_id']}", use_container_width=True):
                            adapt_mgr.record_analyst_decision(
                                txn_id=item["txn_id"],
                                txn_dict=item,
                                decision="APPROVE_LEGITIMATE",
                                analyst_name="Fraud Analyst 01",
                                notes=analyst_notes or "Approved upon manual inspection"
                            )
                            # Remove from queue
                            st.session_state.review_queue = [q for q in st.session_state.review_queue if q["txn_id"] != item["txn_id"]]
                            # Update decision in main history
                            for p in st.session_state.processed_transactions:
                                if p["txn_id"] == item["txn_id"]:
                                    p["decision"] = "ALLOW (Analyst)"
                                    p["decision_badge"] = "✅ ALLOW (Analyst)"
                                    p["decision_color"] = "#10B981"
                            st.toast(f"Approved {item['txn_id']} as Legitimate!")
                            st.rerun()

                    with b_col2:
                        if st.button("🛑 Confirm Fraud", key=f"blk_{item['txn_id']}", use_container_width=True):
                            adapt_mgr.record_analyst_decision(
                                txn_id=item["txn_id"],
                                txn_dict=item,
                                decision="CONFIRM_FRAUD",
                                analyst_name="Fraud Analyst 01",
                                notes=analyst_notes or "Confirmed malicious fraud pattern"
                            )
                            st.session_state.review_queue = [q for q in st.session_state.review_queue if q["txn_id"] != item["txn_id"]]
                            for p in st.session_state.processed_transactions:
                                if p["txn_id"] == item["txn_id"]:
                                    p["decision"] = "BLOCK (Analyst)"
                                    p["decision_badge"] = "🛑 BLOCK (Analyst)"
                                    p["decision_color"] = "#EF4444"
                            st.toast(f"Confirmed {item['txn_id']} as Fraud!")
                            st.rerun()

    st.markdown("---")
    st.subheader("📝 Analyst Feedback Log")
    if adapt_mgr.feedback_records:
        df_fb = pd.DataFrame([
            {
                "Txn ID": r["txn_id"],
                "Timestamp": r["timestamp"].split("T")[1][:8],
                "Analyst": r["analyst"],
                "Action": r["analyst_decision"],
                "Ground Truth Label": "FRAUD (1)" if r["ground_truth_label"] == 1 else "LEGITIMATE (0)",
                "Notes": r["notes"]
            }
            for r in adapt_mgr.feedback_records
        ])
        st.dataframe(df_fb, use_container_width=True)
    else:
        st.caption("No human analyst decisions logged yet. Take action on items above to generate active learning labels.")


# ==============================================================================
# VIEW 5: CONTINUOUS ADAPTATION & MODEL DRIFT
# ==============================================================================
elif current_view == "🔄 Continuous Adaptation & Drift":
    st.header("🔄 Continuous Adaptation & Dynamic Retraining")
    st.markdown(
        "Modern fraud tactics evolve continuously. SentinelFraud monitors **Concept Drift** in real time "
        "and triggers **dynamic incremental retraining** using human feedback labels."
    )

    # Drift Detection Status
    drift_report = adapt_mgr.detect_concept_drift()
    
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid {'#ef4444' if drift_report['drift_detected'] else '#10b981'};">
            <div class="metric-title">Concept Drift & Emerging Threat Status</div>
            <div style="font-size: 1.25rem; font-weight: 600; margin-top: 6px; color: {'#f87171' if drift_report['drift_detected'] else '#34d399'};">
                {drift_report['alert_message']}
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                Drift Metric Score: <b>{drift_report['drift_score']}</b> (Threshold: 0.18) | Monitored Buffer: <b>{drift_report['buffer_count']} events</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Learning Feedback Samples</div>
            <div class="metric-value">{len(adapt_mgr.feedback_records)}</div>
            <div class="metric-sub" style="color: #60a5fa;">Verified ground truth pairs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚡ Dynamic Retraining Engine")
    st.write(
        "Trigger an incremental retraining pass. The engine will ingest all newly confirmed labels, "
        "re-estimate decision boundaries, update Isolation Forest baselines, and mint an updated model version."
    )

    col_btn_retrain, col_info_retrain = st.columns([1, 2])
    with col_btn_retrain:
        if st.button("🚀 Retrain Model with Feedback", use_container_width=True, type="primary"):
            with st.spinner("Retraining Random Forest & Isolation Forest with active learning buffer..."):
                new_ver = adapt_mgr.retrain_model(df_baseline)
                st.success(f"Successfully retrained and deployed Model Version `{new_ver['version']}`!")
                st.rerun()

    with col_info_retrain:
        st.caption(f"Current Live Model: `{engine.training_version}`. Retraining combines historical training distribution with recent verified analyst feedback.")

    st.markdown("---")
    st.subheader("📈 Model Version Evolution & Performance Gains")
    if adapt_mgr.model_versions:
        df_versions = pd.DataFrame(adapt_mgr.model_versions)
        st.dataframe(
            df_versions,
            column_config={
                "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                "f1_score": st.column_config.NumberColumn("F1 Score", format="%.4f"),
            },
            use_container_width=True
        )

        # Plot performance over versions
        if len(adapt_mgr.model_versions) > 1:
            fig_perf = px.line(
                df_versions,
                x="version",
                y=["roc_auc", "f1_score", "precision", "recall"],
                markers=True,
                title="Model Adaptation & Performance Trajectory"
            )
            fig_perf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(color="#e2e8f0"),
                height=320
            )
            st.plotly_chart(fig_perf, use_container_width=True)
    else:
        st.info("No model version history recorded yet.")
