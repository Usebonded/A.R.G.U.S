"""Pattern Recognition Engine: Supervised ML, Isolation Forest, and Graph Fraud Rings."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import networkx as nx
import joblib

FEATURE_NAMES = [
    "log_amount",
    "log_account_age",
    "kyc_inversion",
    "email_domain_risk",
    "prior_disputes",
    "country_mismatch",
    "distance_from_billing_km",
    "travel_velocity_kmh",
    "is_vpn_or_tor",
    "device_risk",
    "is_emulator",
    "time_on_checkout_s",
    "bot_checkout_flag",
    "failed_logins_24h",
    "mouse_behavior_risk",
    "is_prepaid_card",
]


class FeatureExtractor:
    """Transforms raw transaction dictionaries or DataFrames into feature matrices."""

    @staticmethod
    def extract_features(df: pd.DataFrame) -> np.ndarray:
        # Precompute vectorized features
        log_amount = np.log1p(df["amount"].astype(float).clip(lower=0))
        log_account_age = np.log1p(df["account_age_days"].astype(float).clip(lower=0))
        kyc_inversion = (3 - df["kyc_tier"].astype(int)).clip(lower=0)
        email_domain_risk = df["email_domain_risk"].astype(float)
        prior_disputes = df["prior_disputes"].astype(float)
        
        country_mismatch = (
            (df["card_country"] != df["billing_country"]) |
            (df["ip_country"] != df["billing_country"])
        ).astype(int)
        
        distance = df["distance_from_billing_km"].astype(float)
        velocity = df["travel_velocity_kmh"].astype(float)
        is_vpn = df["is_vpn_or_tor"].astype(int)
        device_risk = 1.0 - df["device_trust_score"].astype(float)
        is_emulator = df["is_emulator"].astype(int)
        time_on_checkout = df["time_on_checkout_s"].astype(float)
        bot_checkout_flag = (time_on_checkout < 3.5).astype(int)
        failed_logins = df["failed_logins_24h"].astype(float)
        mouse_risk = df["mouse_behavior_risk"].astype(float)
        is_prepaid = (df["card_type"] == "Prepaid").astype(int)

        features = np.column_stack([
            log_amount,
            log_account_age,
            kyc_inversion,
            email_domain_risk,
            prior_disputes,
            country_mismatch,
            distance,
            velocity,
            is_vpn,
            device_risk,
            is_emulator,
            time_on_checkout,
            bot_checkout_flag,
            failed_logins,
            mouse_risk,
            is_prepaid,
        ])
        return features

    @classmethod
    def extract_single(cls, txn_dict: Dict[str, Any]) -> np.ndarray:
        df = pd.DataFrame([txn_dict])
        return cls.extract_features(df)


class FraudGraphDetector:
    """Detects syndicate fraud rings, shared devices, and multi-account card reuse."""

    def __init__(self):
        self.graph = nx.Graph()
        self.device_users: Dict[str, set] = {}
        self.card_users: Dict[str, set] = {}

    def add_transaction(self, user_id: str, device_id: str, card_id: str, ip_address: str):
        # Update user mapping
        if device_id not in self.device_users:
            self.device_users[device_id] = set()
        self.device_users[device_id].add(user_id)

        if card_id not in self.card_users:
            self.card_users[card_id] = set()
        self.card_users[card_id].add(user_id)

        # Graph nodes and edges
        u_node = f"user:{user_id}"
        d_node = f"device:{device_id}"
        c_node = f"card:{card_id}"
        ip_node = f"ip:{ip_address}"

        self.graph.add_node(u_node, node_type="User")
        self.graph.add_node(d_node, node_type="Device")
        self.graph.add_node(c_node, node_type="Card")
        self.graph.add_node(ip_node, node_type="IP")

        self.graph.add_edge(u_node, d_node)
        self.graph.add_edge(u_node, c_node)
        self.graph.add_edge(u_node, ip_node)

    def analyze_entity(self, user_id: str, device_id: str, card_id: str) -> Dict[str, Any]:
        """Calculates graph risk score and returns shared entity metadata."""
        shared_devices_count = len(self.device_users.get(device_id, set()))
        shared_cards_count = len(self.card_users.get(card_id, set()))

        risk = 0.0
        reasons = []

        if shared_devices_count >= 3:
            risk += 0.55
            reasons.append(f"Device {device_id} recycled across {shared_devices_count} separate accounts")
        elif shared_devices_count == 2:
            risk += 0.25
            reasons.append(f"Device {device_id} shared with 1 other account")

        if shared_cards_count >= 3:
            risk += 0.45
            reasons.append(f"Card {card_id} shared across {shared_cards_count} accounts")
        elif shared_cards_count == 2:
            risk += 0.20
            reasons.append(f"Card {card_id} reused across 2 accounts")

        ring_risk = min(1.0, risk)
        return {
            "graph_risk": ring_risk,
            "shared_device_accounts": shared_devices_count,
            "shared_card_accounts": shared_cards_count,
            "reasons": reasons,
        }

    def get_subgraph_for_user(self, user_id: str, max_depth: int = 2) -> nx.Graph:
        """Returns the ego network around a user for interactive visualization."""
        target = f"user:{user_id}"
        if target not in self.graph:
            sub = nx.Graph()
            sub.add_node(target, node_type="User")
            return sub
        
        nodes = {target}
        frontier = {target}
        for _ in range(max_depth):
            next_frontier = set()
            for n in frontier:
                for neighbor in self.graph.neighbors(n):
                    if neighbor not in nodes:
                        nodes.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return self.graph.subgraph(nodes).copy()


class PatternRecognitionEngine:
    """Hybrid pattern recognition combining Supervised Ensemble, Isolation Forest, and Graph Networks."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.supervised_model = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42
        )
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        self.graph_detector = FraudGraphDetector()
        self.is_trained = False
        self.training_version = "v1.0"

    def train(self, df_train: pd.DataFrame, version: str = "v1.0"):
        """Trains both the supervised model and unsupervised Isolation Forest on historical data."""
        X_raw = FeatureExtractor.extract_features(df_train)
        y = df_train["is_fraud"].values.astype(int)

        self.scaler.fit(X_raw)
        X_scaled = self.scaler.transform(X_raw)

        self.supervised_model.fit(X_scaled, y)
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True
        self.training_version = version

        # Populate graph with historical entities
        for _, row in df_train.iterrows():
            self.graph_detector.add_transaction(
                user_id=row["user_id"],
                device_id=row["device_id"],
                card_id=row["card_id"],
                ip_address=row["ip_address"]
            )

    def predict(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Infers supervised probability, unsupervised anomaly score, and graph link risk."""
        if not self.is_trained:
            raise RuntimeError("PatternRecognitionEngine must be trained before inference.")

        # Real-time graph registration
        self.graph_detector.add_transaction(
            user_id=txn_dict["user_id"],
            device_id=txn_dict["device_id"],
            card_id=txn_dict["card_id"],
            ip_address=txn_dict["ip_address"]
        )

        X_raw = FeatureExtractor.extract_single(txn_dict)
        X_scaled = self.scaler.transform(X_raw)

        # 1. Supervised probability
        supervised_prob = float(self.supervised_model.predict_proba(X_scaled)[0, 1])

        # 2. Isolation Forest score
        # decision_function gives negative values for outliers, positive for inliers
        raw_if_score = float(self.isolation_forest.decision_function(X_scaled)[0])
        # Map raw IF score (approx -0.3 to +0.2) to 0.0 - 1.0 anomaly severity
        # lower raw_if_score => higher anomaly
        anomaly_score = float(np.clip(1.0 - (raw_if_score + 0.25) / 0.45, 0.0, 1.0))

        # 3. Graph link analysis
        graph_meta = self.graph_detector.analyze_entity(
            user_id=txn_dict["user_id"],
            device_id=txn_dict["device_id"],
            card_id=txn_dict["card_id"]
        )

        return {
            "supervised_probability": supervised_prob,
            "anomaly_score": anomaly_score,
            "graph_risk": graph_meta["graph_risk"],
            "graph_meta": graph_meta,
            "features_raw": dict(zip(FEATURE_NAMES, X_raw[0])),
        }
