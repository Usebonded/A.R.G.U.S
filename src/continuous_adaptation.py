"""Continuous Adaptation & Feedback Loop Module.

Manages:
1. Analyst Human-in-the-Loop Feedback Queue
2. Emerging Concept Drift & Attack Vector Monitoring
3. Dynamic Incremental Model Retraining
4. Model Versioning & Performance Evolution Tracking
"""

import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.pattern_recognition import PatternRecognitionEngine, FeatureExtractor
from src.synthetic_data import generate_dataset


class ContinuousAdaptationManager:
    """Orchestrates human feedback, drift detection, and automated model updates."""

    def __init__(self, engine: PatternRecognitionEngine):
        self.engine = engine
        self.feedback_records: List[Dict[str, Any]] = []
        self.recent_stream_buffer: List[Dict[str, Any]] = []
        self.model_versions: List[Dict[str, Any]] = []
        self.version_counter = 1

    def record_analyst_decision(
        self,
        txn_id: str,
        txn_dict: Dict[str, Any],
        decision: str,  # "CONFIRM_FRAUD" or "APPROVE_LEGITIMATE"
        analyst_name: str = "Fraud Analyst 01",
        notes: str = ""
    ):
        """Logs an analyst verification from the Manual Review queue."""
        is_fraud = 1 if decision == "CONFIRM_FRAUD" else 0
        record = {
            "txn_id": txn_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "analyst": analyst_name,
            "analyst_decision": decision,
            "ground_truth_label": is_fraud,
            "notes": notes,
            "transaction_data": txn_dict
        }
        self.feedback_records.append(record)
        return record

    def add_stream_observation(self, txn_dict: Dict[str, Any]):
        """Buffers recent stream observations for rolling drift analysis."""
        self.recent_stream_buffer.append(txn_dict)
        if len(self.recent_stream_buffer) > 300:
            self.recent_stream_buffer.pop(0)

    def detect_concept_drift(self) -> Dict[str, Any]:
        """Analyzes recent transactions for statistical drift or emerging attack vectors."""
        if len(self.recent_stream_buffer) < 20:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "rolling_fraud_ratio": 0.0,
                "alert_message": "Collecting stream telemetry (need >= 20 events)...",
                "emerging_factors": []
            }

        df_recent = pd.DataFrame(self.recent_stream_buffer)
        
        # Calculate recent metrics
        vpn_ratio = df_recent["is_vpn_or_tor"].mean()
        high_vel_ratio = (df_recent["travel_velocity_kmh"] > 500).mean()
        bot_checkout_ratio = (df_recent["time_on_checkout_s"] < 3.5).mean()
        disposable_email_ratio = (df_recent["email_domain_risk"] > 0.7).mean()

        drift_score = (vpn_ratio * 0.3 + high_vel_ratio * 0.35 + bot_checkout_ratio * 0.2 + disposable_email_ratio * 0.15)
        drift_detected = bool(drift_score > 0.18 or high_vel_ratio > 0.12)

        emerging_factors = []
        if high_vel_ratio > 0.10:
            emerging_factors.append(f"Impossible travel spike ({high_vel_ratio*100:.1f}% of recent txns)")
        if vpn_ratio > 0.25:
            emerging_factors.append(f"Anonymizer/VPN surge ({vpn_ratio*100:.1f}% of recent traffic)")
        if bot_checkout_ratio > 0.10:
            emerging_factors.append(f"Automated headless checkout burst ({bot_checkout_ratio*100:.1f}%)")

        alert = "Normal distribution (no critical drift)"
        if drift_detected:
            alert = f"⚠️ Concept drift detected! Emerging fraud vectors: {', '.join(emerging_factors)}"

        return {
            "drift_detected": drift_detected,
            "drift_score": round(float(drift_score), 3),
            "alert_message": alert,
            "emerging_factors": emerging_factors,
            "buffer_count": len(self.recent_stream_buffer)
        }

    def retrain_model(self, base_df: pd.DataFrame) -> Dict[str, Any]:
        """Dynamically retrains the engine using base dataset + human feedback."""
        self.version_counter += 1
        new_version_tag = f"v1.{self.version_counter - 1}"

        # Convert feedback records to DataFrame if any
        if self.feedback_records:
            feedback_rows = []
            for r in self.feedback_records:
                row = dict(r["transaction_data"])
                row["is_fraud"] = r["ground_truth_label"]
                feedback_rows.append(row)
            df_feedback = pd.DataFrame(feedback_rows)
            # Combine base with feedback with weighting
            augmented_df = pd.concat([base_df, df_feedback], ignore_index=True)
        else:
            augmented_df = base_df.copy()

        # Retrain the engine
        self.engine.train(augmented_df, version=new_version_tag)

        # Evaluate performance on test slice
        test_df = generate_dataset(n_samples=400, fraud_ratio=0.08, seed=99 + self.version_counter)
        X_test = FeatureExtractor.extract_features(test_df)
        X_test_scaled = self.engine.scaler.transform(X_test)
        y_true = test_df["is_fraud"].values.astype(int)

        y_prob = self.engine.supervised_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = float(roc_auc_score(y_true, y_prob))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        version_info = {
            "version": new_version_tag,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC"),
            "training_samples": len(augmented_df),
            "human_feedback_samples": len(self.feedback_records),
            "roc_auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }
        self.model_versions.append(version_info)
        return version_info
