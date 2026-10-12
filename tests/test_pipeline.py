"""Comprehensive Pipeline Test for Fraud Detection System."""

import sys
import os

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import TransactionRecord
from src.synthetic_data import generate_single_transaction, generate_dataset
from src.pattern_recognition import PatternRecognitionEngine, FeatureExtractor
from src.risk_engine import RiskEngine
from src.continuous_adaptation import ContinuousAdaptationManager


def test_synthetic_data_generation():
    print("Testing synthetic data generation...")
    # Normal transaction
    t_normal = generate_single_transaction(txn_index=1, force_attack="Normal")
    assert isinstance(t_normal, TransactionRecord)
    assert t_normal.is_fraud == 0
    assert t_normal.attack_type == "Normal"

    # Impossible Travel attack
    t_travel = generate_single_transaction(txn_index=2, force_attack="Impossible Travel")
    assert t_travel.is_fraud == 1
    assert t_travel.travel_velocity_kmh > 800

    # Dataset generation
    df = generate_dataset(n_samples=200, fraud_ratio=0.10, seed=42)
    assert len(df) == 200
    assert df["is_fraud"].sum() == 20
    print("✓ Synthetic data generation passed.")
    return df


def test_pattern_recognition_and_scoring(df_train):
    print("Testing pattern recognition engine training & inference...")
    engine = PatternRecognitionEngine()
    engine.train(df_train)
    assert engine.is_trained is True

    # Test legitimate transaction inference
    t_legit = generate_single_transaction(txn_index=10, force_attack="Normal").to_dict()
    pred_legit = engine.predict(t_legit)
    eval_legit = RiskEngine.evaluate(t_legit, pred_legit)

    print(f"Normal transaction risk score: {eval_legit['risk_score']} ({eval_legit['decision']})")
    assert 0 <= eval_legit["risk_score"] <= 100
    assert eval_legit["decision"] in ["ALLOW", "MANUAL REVIEW"]

    # Test extreme attack transaction inference
    t_attack = generate_single_transaction(txn_index=11, force_attack="Impossible Travel").to_dict()
    pred_attack = engine.predict(t_attack)
    eval_attack = RiskEngine.evaluate(t_attack, pred_attack)

    print(f"Impossible Travel risk score: {eval_attack['risk_score']} ({eval_attack['decision']})")
    assert eval_attack["risk_score"] >= 50
    assert eval_attack["decision"] in ["MANUAL REVIEW", "BLOCK"]
    assert len(eval_attack["drivers"]) > 0

    print("✓ Pattern recognition & risk scoring passed.")
    return engine


def test_continuous_adaptation(engine, df_train):
    print("Testing continuous adaptation and dynamic retraining...")
    adapt_mgr = ContinuousAdaptationManager(engine)

    # Simulate analyst feedback
    t_flagged = generate_single_transaction(txn_index=99, force_attack="Credential Stuffing ATO").to_dict()
    adapt_mgr.record_analyst_decision(
        txn_id=t_flagged["txn_id"],
        txn_dict=t_flagged,
        decision="CONFIRM_FRAUD",
        analyst_name="Test Lead",
        notes="Confirmed brute force authentication attack from VPN node"
    )
    assert len(adapt_mgr.feedback_records) == 1

    # Simulate drift buffer
    for i in range(25):
        sample = generate_single_transaction(txn_index=200+i, force_attack="Impossible Travel").to_dict()
        adapt_mgr.add_stream_observation(sample)

    drift_report = adapt_mgr.detect_concept_drift()
    print("Drift report:", drift_report)
    assert drift_report["drift_detected"] is True

    # Retrain
    version_info = adapt_mgr.retrain_model(df_train)
    print("Retrained model version info:", version_info)
    assert version_info["version"] == "v1.1"
    assert version_info["human_feedback_samples"] == 1
    assert version_info["roc_auc"] > 0.70

    print("✓ Continuous adaptation passed.")


if __name__ == "__main__":
    df = test_synthetic_data_generation()
    engine = test_pattern_recognition_and_scoring(df)
    test_continuous_adaptation(engine, df)
    print("\nALL TEST SUITES PASSED SUCCESSFULLY!")
