import pandas as pd

def get_data():
    # Load CSVs
    launches = pd.read_csv("launches.csv")
    cores = pd.read_csv("cores.csv")
    capsules = pd.read_csv("capsules.csv")
    payloads = pd.read_csv("payloads.csv")
    ships = pd.read_csv("ships.csv")
    launchpads = pd.read_csv("launchpads.csv")
    rockets = pd.read_csv("rockets.csv")

    # ====================================================
    # REMOVE CONFLICTING NAME COLUMNS (VERY IMPORTANT)
    # ====================================================
    
    # Keep only launch_name - drop all other .name fields
    launches = launches.rename(columns={"name": "launch_name"})
    
    for df in [payloads, rockets, ships]:
        if "name" in df.columns:
            df.drop(columns=["name"], inplace=True)

    # ====================================================
    # RENAME COLUMNS TO MATCH MODEL EXPECTATIONS
    # ====================================================

    # Launchpads
    launchpads = launchpads.rename(columns={
        "status": "launchpad_status",
        "region": "launchpad_region"
    })

    # Cores
    cores = cores.rename(columns={
        "reuse_count": "core_reuse_count",
        "block": "core_block",
        "rtls_attempts": "core_rtls_attempts",
        "rtls_landings": "core_rtls_landings",
        "asds_attempts": "core_asds_attempts",
        "asds_landings": "core_asds_landings",
        "status": "core_status"
    })

    # Capsules
    capsules = capsules.rename(columns={
        "reuse_count": "capsule_reuse_count",
        "status": "capsule_status"
    })

    # Payloads
    payloads = payloads.rename(columns={
        "mass_kg": "payload_mass_kg",
        "type": "payload_type",
        "orbit": "payload_orbit",
        "reference_system": "payload_ref_sys",
        "regime": "payload_regime"
    })

    # Ships
    ships = ships.rename(columns={
        "active": "ship_active",
        "type": "ship_type"
    })

    # Rockets
    rockets = rockets.rename(columns={
        "active": "rocket_active",
        "type": "rocket_type"
    })

    # ====================================================
    # MERGE TABLES IN SEQUENCE
    # ====================================================

    df = launches.merge(launchpads, on="launchpad_id", how="left")
    df = df.merge(cores, on="core_id", how="left")
    df = df.merge(capsules, on="capsule_id", how="left")
    df = df.merge(payloads, on="payload_id", how="left")
    df = df.merge(ships, on="ship_id", how="left")
    df = df.merge(rockets, on="rocket_id", how="left")

    return df
