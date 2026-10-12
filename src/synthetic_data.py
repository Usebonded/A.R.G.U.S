"""Synthetic Data Generator & Real-Time Stream Simulator for Fraud Detection."""

import random
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.schema import TransactionRecord

# Reference pools for realistic data generation
TRUSTED_EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "icloud.com", "corp.com"]
DISPOSABLE_EMAIL_DOMAINS = ["tempmail.ninja", "dispostable.com", "10minutemail.net", "trashmail.org"]

CARD_BRANDS = ["Visa", "Mastercard", "Amex", "Discover"]
CARD_TYPES = ["Credit", "Debit", "Prepaid"]

COUNTRIES = [
    {"code": "US", "city": "New York", "lat": 40.7128, "lon": -74.0060},
    {"code": "US", "city": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"code": "GB", "city": "London", "lat": 51.5074, "lon": -0.1278},
    {"code": "DE", "city": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
    {"code": "SG", "city": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"code": "JP", "city": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"code": "BR", "city": "Sao Paulo", "lat": -23.5505, "lon": -46.6333},
    {"code": "NG", "city": "Lagos", "lat": 6.5244, "lon": 3.3792},
    {"code": "RU", "city": "Moscow", "lat": 55.7558, "lon": 37.6173},
]

RESIDENTIAL_ASNS = ["Comcast-Cable-US", "Verizon-Fios-US", "Deutsche-Telekom-DE", "Vodafone-GB", "Singtel-SG", "NTT-JP"]
HOSTING_PROXY_ASNS = ["DigitalOcean-Cloud-US", "OVH-Hosting-FR", "Mullvad-VPN-SE", "NordVPN-Exit-NL", "Tor-Exit-DE", "AWS-Datacenter-US"]

# Fraud ring shared entities pool
SHARED_RING_DEVICES = ["DEV-RING-ALPHA-89", "DEV-RING-BETA-42"]
SHARED_RING_CARDS = ["CARD-STOLEN-9901", "CARD-STOLEN-4412"]


def _random_geo_jitter(base_lat: float, base_lon: float, max_offset_km: float = 20.0):
    # 1 degree approx 111 km
    lat_offset = (random.random() - 0.5) * (max_offset_km / 111.0)
    lon_offset = (random.random() - 0.5) * (max_offset_km / 111.0)
    return round(base_lat + lat_offset, 4), round(base_lon + lon_offset, 4)


def generate_single_transaction(
    txn_index: int = 1,
    force_attack: Optional[str] = None,
    rng: Optional[random.Random] = None
) -> TransactionRecord:
    """Generates a single realistic transaction. If force_attack is set, generates that attack pattern."""
    r = rng or random
    
    # Determine if legitimate or fraud
    attack_types = [
        "Impossible Travel",
        "Credential Stuffing ATO",
        "Card Testing Bot",
        "Syndicate Fraud Ring",
    ]
    
    if force_attack:
        attack = force_attack
        is_fraud = 1 if force_attack != "Normal" else 0
    else:
        # Default 4% fraud rate if purely stochastic
        if r.random() < 0.04:
            is_fraud = 1
            attack = r.choice(attack_types)
        else:
            is_fraud = 0
            attack = "Normal"

    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=r.randint(0, 86400 * 7))
    txn_id = f"TXN-{100000 + txn_index}"
    user_id = f"USR-{r.randint(1000, 9999)}"
    card_id = f"CARD-{r.randint(1000, 9999)}"
    device_id = f"DEV-{r.randint(10000, 99999)}"

    # Default legitimate attributes
    account_age = r.randint(30, 1200)
    kyc_tier = r.choice([1, 2, 2, 3])
    email_domain_risk = round(r.uniform(0.01, 0.15), 3)
    prior_disputes = 0 if r.random() > 0.05 else 1
    
    card_brand = r.choice(CARD_BRANDS)
    card_type = r.choice(["Credit", "Credit", "Debit"])
    
    home_country = r.choice(COUNTRIES[:4]) # US, GB, DE
    card_country = home_country["code"]
    billing_country = home_country["code"]
    
    amount = round(float(np.random.lognormal(mean=3.8, sigma=1.0)), 2)  # ~$20 - $250
    amount = max(5.0, min(amount, 2500.0))
    currency = "USD"
    
    ip_country_info = home_country
    ip_lat, ip_lon = _random_geo_jitter(home_country["lat"], home_country["lon"], max_offset_km=15)
    distance_billing = round(r.uniform(0.5, 45.0), 1)
    time_since_last_txn = round(r.uniform(120.0, 4320.0), 1) # minutes
    travel_velocity = round(distance_billing / max(time_since_last_txn / 60.0, 0.1), 1)
    
    asn_org = r.choice(RESIDENTIAL_ASNS)
    is_vpn = False
    device_trust = round(r.uniform(0.75, 0.99), 3)
    is_emulator = False
    
    session_duration = round(r.uniform(35.0, 240.0), 1)
    failed_logins = 0 if r.random() > 0.06 else 1
    checkout_time = round(r.uniform(12.0, 65.0), 1)
    mouse_risk = round(r.uniform(0.05, 0.28), 3)

    # Inject specific attack archetype anomalies
    if is_fraud:
        if attack == "Impossible Travel":
            # Current IP is thousands of miles away within minutes of previous txn
            foreign_country = r.choice(COUNTRIES[4:]) # SG, JP, BR, NG, RU
            ip_country_info = foreign_country
            ip_lat, ip_lon = _random_geo_jitter(foreign_country["lat"], foreign_country["lon"], max_offset_km=30)
            distance_billing = round(r.uniform(4500.0, 11000.0), 1)
            time_since_last_txn = round(r.uniform(10.0, 45.0), 1)  # only 10-45 mins ago!
            travel_velocity = round(distance_billing / (time_since_last_txn / 60.0), 1)  # > 2,000 km/h!
            amount = round(r.uniform(450.0, 2100.0), 2)
            asn_org = r.choice(HOSTING_PROXY_ASNS)
            is_vpn = True
            device_trust = round(r.uniform(0.2, 0.5), 3)

        elif attack == "Credential Stuffing ATO":
            # Account takeover: high failed logins, brand new untrusted device, fast checkout, high amount
            account_age = r.randint(200, 800)  # Established victim account
            failed_logins = r.randint(4, 11)   # Repeated password failures
            session_duration = round(r.uniform(8.0, 25.0), 1)  # In and out fast
            checkout_time = round(r.uniform(4.0, 9.0), 1)
            amount = round(r.uniform(850.0, 3400.0), 2)
            device_trust = round(r.uniform(0.05, 0.25), 3)    # Completely unverified device
            asn_org = r.choice(HOSTING_PROXY_ASNS)
            is_vpn = True
            mouse_risk = round(r.uniform(0.65, 0.95), 3)       # Scripted or tense cadence
            kyc_tier = 1

        elif attack == "Card Testing Bot":
            # Automated bot probing stolen card validity with micro-transactions
            account_age = r.randint(0, 2)       # Brand new burner account
            kyc_tier = 0
            email_domain_risk = round(r.uniform(0.85, 0.99), 3)
            card_type = "Prepaid"
            amount = round(r.uniform(0.99, 4.99), 2)  # Micro-charge test
            checkout_time = round(r.uniform(0.8, 2.8), 1)  # Sub-3 second headless bot
            session_duration = round(r.uniform(2.0, 8.0), 1)
            mouse_risk = round(r.uniform(0.88, 0.99), 3)
            is_emulator = True
            is_vpn = True
            asn_org = r.choice(HOSTING_PROXY_ASNS)
            device_trust = round(r.uniform(0.01, 0.15), 3)

        elif attack == "Syndicate Fraud Ring":
            # Multiple accounts recycled over identical device fingerprint & card
            device_id = r.choice(SHARED_RING_DEVICES)
            card_id = r.choice(SHARED_RING_CARDS)
            account_age = r.randint(1, 14)
            prior_disputes = r.randint(2, 5)
            card_type = "Credit"
            amount = round(r.uniform(600.0, 2800.0), 2)
            card_country = "NG"
            billing_country = "US"
            distance_billing = round(r.uniform(2500.0, 8000.0), 1)
            is_vpn = True
            asn_org = "DigitalOcean-Cloud-US"
            device_trust = round(r.uniform(0.20, 0.45), 3)
            mouse_risk = round(r.uniform(0.40, 0.75), 3)

    return TransactionRecord(
        txn_id=txn_id,
        timestamp=now.isoformat(),
        user_id=user_id,
        account_age_days=account_age,
        kyc_tier=kyc_tier,
        email_domain_risk=email_domain_risk,
        prior_disputes=prior_disputes,
        card_id=card_id,
        card_brand=card_brand,
        card_type=card_type,
        card_country=card_country,
        billing_country=billing_country,
        amount=amount,
        currency=currency,
        ip_country=ip_country_info["code"],
        ip_city=ip_country_info["city"],
        ip_lat=ip_lat,
        ip_lon=ip_lon,
        distance_from_billing_km=distance_billing,
        time_since_last_txn_min=time_since_last_txn,
        travel_velocity_kmh=travel_velocity,
        device_id=device_id,
        ip_address=f"{r.randint(11, 215)}.{r.randint(20, 250)}.{r.randint(1, 254)}.{r.randint(1, 254)}",
        asn_org=asn_org,
        is_vpn_or_tor=is_vpn,
        device_trust_score=device_trust,
        is_emulator=is_emulator,
        session_duration_s=session_duration,
        failed_logins_24h=failed_logins,
        time_on_checkout_s=checkout_time,
        mouse_behavior_risk=mouse_risk,
        is_fraud=is_fraud,
        attack_type=attack
    )


def generate_dataset(n_samples: int = 2000, fraud_ratio: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Generates a comprehensive historical training and benchmark dataset."""
    rng = random.Random(seed)
    np.random.seed(seed)
    records = []
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate normal records
    for i in range(n_normal):
        rec = generate_single_transaction(txn_index=i+1, force_attack="Normal", rng=rng)
        records.append(rec.to_dict())
        
    # Generate fraud records with varied archetypes
    archetypes = [
        "Impossible Travel",
        "Credential Stuffing ATO",
        "Card Testing Bot",
        "Syndicate Fraud Ring",
    ]
    for j in range(n_fraud):
        attack = archetypes[j % len(archetypes)]
        rec = generate_single_transaction(txn_index=n_normal+j+1, force_attack=attack, rng=rng)
        records.append(rec.to_dict())
        
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
