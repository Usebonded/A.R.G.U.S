"""Calibrated Risk Scoring Engine & Explainable AI (XAI) Decision System."""

from typing import Dict, Any, List, Tuple
import numpy as np


class RiskEngine:
    """Computes a calibrated 0–100 Composite Risk Score, assigns action tiers,

    and extracts explainable decision drivers.
    """

    # Thresholds
    ALLOW_THRESHOLD = 30
    BLOCK_THRESHOLD = 75

    @classmethod
    def evaluate(cls, txn: Dict[str, Any], pattern_output: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates calibrated composite score, dimensional sub-scores, and XAI drivers."""
        sup_prob = pattern_output["supervised_probability"]  # 0.0 to 1.0
        anom_score = pattern_output["anomaly_score"]        # 0.0 to 1.0
        graph_risk = pattern_output["graph_risk"]            # 0.0 to 1.0

        # Heuristic rules and risk driver tracking
        drivers: List[Dict[str, Any]] = []
        rule_penalties = 0.0

        # Rule 1: Impossible Travel Velocity
        velocity = float(txn.get("travel_velocity_kmh", 0))
        if velocity > 800:
            impact = 35.0
            rule_penalties += impact
            drivers.append({
                "factor": "Impossible Travel Velocity",
                "impact": f"+{int(impact)} pts",
                "severity": "CRITICAL",
                "detail": f"Speed {velocity:,.0f} km/h exceeds maximum jet travel capability"
            })
        elif velocity > 250:
            impact = 18.0
            rule_penalties += impact
            drivers.append({
                "factor": "High Travel Velocity",
                "impact": f"+{int(impact)} pts",
                "severity": "MEDIUM",
                "detail": f"Fast transit velocity of {velocity:,.0f} km/h detected between checkouts"
            })

        # Rule 2: Syndicate Graph Connection
        shared_devices = pattern_output.get("graph_meta", {}).get("shared_device_accounts", 1)
        shared_cards = pattern_output.get("graph_meta", {}).get("shared_card_accounts", 1)
        if shared_devices >= 3:
            impact = 30.0
            rule_penalties += impact
            drivers.append({
                "factor": "Syndicate Device Recycling",
                "impact": f"+{int(impact)} pts",
                "severity": "CRITICAL",
                "detail": f"Hardware device fingerprint shared across {shared_devices} distinct user accounts"
            })
        if shared_cards >= 3:
            impact = 25.0
            rule_penalties += impact
            drivers.append({
                "factor": "Multi-Account Card Sharing",
                "impact": f"+{int(impact)} pts",
                "severity": "HIGH",
                "detail": f"Payment card reused across {shared_cards} accounts in graph topology"
            })

        # Rule 3: Automated Bot Checkout Cadence
        time_checkout = float(txn.get("time_on_checkout_s", 30))
        mouse_risk = float(txn.get("mouse_behavior_risk", 0.1))
        if time_checkout < 3.0 and mouse_risk > 0.8:
            impact = 28.0
            rule_penalties += impact
            drivers.append({
                "factor": "Scripted Headless Bot Cadence",
                "impact": f"+{int(impact)} pts",
                "severity": "HIGH",
                "detail": f"Sub-3 second checkout ({time_checkout:.1f}s) with robotic telemetry ({mouse_risk:.2f})"
            })

        # Rule 4: Credential Attack & Recent Failed Logins
        failed_logins = int(txn.get("failed_logins_24h", 0))
        if failed_logins >= 4:
            impact = 22.0
            rule_penalties += impact
            drivers.append({
                "factor": "Recent Authentication Bursts",
                "impact": f"+{int(impact)} pts",
                "severity": "HIGH",
                "detail": f"{failed_logins} failed login attempts registered in last 24 hours"
            })

        # Rule 5: VPN/Proxy with High-Value Transaction
        is_vpn = bool(txn.get("is_vpn_or_tor", False))
        amount = float(txn.get("amount", 0))
        if is_vpn and amount > 800:
            impact = 20.0
            rule_penalties += impact
            drivers.append({
                "factor": "High-Value Transaction via Anonymizer",
                "impact": f"+{int(impact)} pts",
                "severity": "MEDIUM",
                "detail": f"${amount:,.2f} order routed via proxy/VPN node ({txn.get('asn_org', 'Unknown')})"
            })
        elif is_vpn:
            impact = 10.0
            rule_penalties += impact
            drivers.append({
                "factor": "Proxy/VPN Network Exit",
                "impact": f"+{int(impact)} pts",
                "severity": "LOW",
                "detail": f"Traffic routed through {txn.get('asn_org', 'Proxy network')}"
            })

        # Rule 6: Disposable Email Domain & Low KYC
        email_risk = float(txn.get("email_domain_risk", 0))
        kyc = int(txn.get("kyc_tier", 2))
        if email_risk > 0.8 and kyc == 0:
            impact = 22.0
            rule_penalties += impact
            drivers.append({
                "factor": "Disposable Email & Unverified KYC",
                "impact": f"+{int(impact)} pts",
                "severity": "HIGH",
                "detail": "Ephemeral throwaway email domain with KYC Tier 0 status"
            })

        # Compute Composite Weighted Risk Score
        # ML Base (45%), Anomaly Base (25%), Graph Base (15%), Rules (15% scaled)
        ml_points = sup_prob * 45.0
        anom_points = anom_score * 25.0
        graph_points = graph_risk * 15.0
        rule_points = min(35.0, rule_penalties * 0.4)

        raw_score = ml_points + anom_points + graph_points + rule_points

        # Clamp between 0 and 100
        composite_score = int(np.clip(round(raw_score), 0, 100))

        # Assign Automated Action Tier
        if composite_score < cls.ALLOW_THRESHOLD:
            decision = "ALLOW"
            decision_color = "#10B981"  # Emerald Green
            decision_badge = "✅ ALLOW"
            if not drivers:
                drivers.append({
                    "factor": "Normal Behavioral Patterns",
                    "impact": "Baseline",
                    "severity": "LOW",
                    "detail": "Verified identity, trusted device, and local geolocation signals"
                })
        elif composite_score < cls.BLOCK_THRESHOLD:
            decision = "MANUAL REVIEW"
            decision_color = "#F59E0B"  # Amber Warning
            decision_badge = "⚠️ MANUAL REVIEW"
        else:
            decision = "BLOCK"
            decision_color = "#EF4444"  # Crimson Block
            decision_badge = "🛑 BLOCK"

        # Calculate Dimensional Sub-scores (0 to 100 each for radar chart)
        # 1. Identity Score
        acc_age = float(txn.get("account_age_days", 100))
        acc_age_penalty = max(0, 35 - acc_age * 0.5)
        id_score = int(np.clip(
            email_risk * 40 + (3 - kyc) * 15 + acc_age_penalty + int(txn.get("prior_disputes", 0)) * 20,
            5, 100
        ))

        # 2. Payment Score
        is_foreign = 25 if txn.get("card_country") != txn.get("billing_country") else 0
        is_prepaid = 20 if txn.get("card_type") == "Prepaid" else 0
        amt_penalty = min(40, (amount / 2000.0) * 40)
        pay_score = int(np.clip(is_foreign + is_prepaid + amt_penalty + sup_prob * 30, 5, 100))

        # 3. Location Score
        dist = float(txn.get("distance_from_billing_km", 10))
        dist_factor = min(40, (dist / 3000.0) * 40)
        vel_factor = min(50, (velocity / 700.0) * 50)
        loc_score = int(np.clip(dist_factor + vel_factor + (15 if txn.get("ip_country") != txn.get("billing_country") else 0), 5, 100))

        # 4. Network & Device Score
        dev_trust = float(txn.get("device_trust_score", 0.9))
        dev_risk = (1.0 - dev_trust) * 45
        vpn_risk = 30 if is_vpn else 0
        emu_risk = 25 if txn.get("is_emulator", False) else 0
        ring_mult = 20 if shared_devices >= 2 else 0
        net_score = int(np.clip(dev_risk + vpn_risk + emu_risk + ring_mult, 5, 100))

        # 5. Session Score
        sess_score = int(np.clip(
            mouse_risk * 40 +
            (failed_logins * 12) +
            (35 if time_checkout < 3.5 else 0),
            5, 100
        ))

        dimension_breakdown = {
            "Identity": id_score,
            "Payment": pay_score,
            "Location": loc_score,
            "Network": net_score,
            "Session": sess_score,
        }

        return {
            "risk_score": composite_score,
            "decision": decision,
            "decision_color": decision_color,
            "decision_badge": decision_badge,
            "drivers": drivers,
            "dimension_breakdown": dimension_breakdown,
            "components": {
                "supervised_ml_pts": round(ml_points, 1),
                "anomaly_pts": round(anom_points, 1),
                "graph_pts": round(graph_points, 1),
                "rule_pts": round(rule_points, 1),
            }
        }
