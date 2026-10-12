"""Multi-Entity Transaction Schema for Fraud Detection System.

Captures multimodal signals across:
1. Identity
2. Payment Method
3. Location & Velocity
4. Network & Device
5. Session & Behavioral Telemetry
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import datetime
import uuid


@dataclass
class TransactionRecord:
    # Transaction Metadata
    txn_id: str
    timestamp: str  # ISO timestamp
    
    # 1. Identity Signals
    user_id: str
    account_age_days: int
    kyc_tier: int  # 0: Unverified, 1: Basic (email/phone), 2: ID Verified, 3: Enhanced/Biometric
    email_domain_risk: float  # 0.0 (corporate/trusted) to 1.0 (disposable/tempmail)
    prior_disputes: int
    
    # 2. Payment Method Signals
    card_id: str
    card_brand: str  # Visa, Mastercard, Amex, Discover
    card_type: str  # Credit, Debit, Prepaid
    card_country: str
    billing_country: str
    amount: float
    currency: str
    
    # 3. Location & Velocity Signals
    ip_country: str
    ip_city: str
    ip_lat: float
    ip_lon: float
    distance_from_billing_km: float
    time_since_last_txn_min: float
    travel_velocity_kmh: float
    
    # 4. Network & Device Signals
    device_id: str
    ip_address: str
    asn_org: str
    is_vpn_or_tor: bool
    device_trust_score: float  # 0.0 (suspicious emulator) to 1.0 (known trusted)
    is_emulator: bool
    
    # 5. Session & Behavioral Telemetry
    session_duration_s: float
    failed_logins_24h: int
    time_on_checkout_s: float
    mouse_behavior_risk: float  # 0.0 (natural curve) to 1.0 (robotic/bot)

    # Optional Ground Truth or Simulation Meta
    is_fraud: Optional[int] = None
    attack_type: Optional[str] = "Normal"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionRecord":
        # Create a shallow copy to prevent modifying caller's dict
        clean_data = dict(data)
        if "txn_id" not in clean_data or not clean_data["txn_id"]:
            clean_data["txn_id"] = f"TXN-{uuid.uuid4().hex[:8].upper()}"
        if "timestamp" not in clean_data or not clean_data["timestamp"]:
            clean_data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return cls(**clean_data)


# Standard numerical & categorical columns used in feature representation
CATEGORICAL_COLS = [
    "card_brand",
    "card_type",
    "card_country",
    "billing_country",
    "ip_country",
    "asn_org",
]

NUMERICAL_COLS = [
    "account_age_days",
    "kyc_tier",
    "email_domain_risk",
    "prior_disputes",
    "amount",
    "distance_from_billing_km",
    "time_since_last_txn_min",
    "travel_velocity_kmh",
    "is_vpn_or_tor",
    "device_trust_score",
    "is_emulator",
    "session_duration_s",
    "failed_logins_24h",
    "time_on_checkout_s",
    "mouse_behavior_risk",
]
