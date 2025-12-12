import pandas as pd


def get_data():
    # -------------------------------
    # Load CSVs (must be in same folder)
    # -------------------------------
    launches = pd.read_csv("launches.csv")
    cores = pd.read_csv("cores.csv")
    capsules = pd.read_csv("capsules.csv")
    payloads = pd.read_csv("payloads.csv")
    ships = pd.read_csv("ships.csv")
    launchpads = pd.read_csv("launchpads.csv")
    rockets = pd.read_csv("rockets.csv")

    # -------------------------------
    # Clean / rename columns
    # -------------------------------

    # Keep launch name but avoid name clashes
    launches = launches.rename(columns={"name": "launch_name"})

    # Drop 'name' columns we don’t need (to avoid merge conflicts)
    for df in (payloads, ships, rockets):
        if "name" in df.columns:
            df.drop(columns=["name"], inplace=True)

    # Launchpads
    launchpads = launchpads.rename(
        columns={
            "status": "launchpad_status",
            "region": "launchpad_region",
        }
    )

    # Cores
    cores = cores.rename(
        columns={
            "status": "core_status",
            "reuse_count": "core_reuse_count",
            "block": "core_block",
            "rtls_attempts": "core_rtls_attempts",
            "rtls_landings": "core_rtls_landings",
            "asds_attempts": "core_asds_attempts",
            "asds_landings": "core_asds_landings",
        }
    )

    # Capsules
    capsules = capsules.rename(
        columns={
            "status": "capsule_status",
            "reuse_count": "capsule_reuse_count",
        }
    )

    # Payloads
    payloads = payloads.rename(
        columns={
            "type": "payload_type",
            "mass_kg": "payload_mass_kg",
            "orbit": "payload_orbit",
            "reference_system": "payload_ref_sys",
            "regime": "payload_regime",
        }
    )

    # Ships
    ships = ships.rename(
        columns={
            "active": "ship_active",
            "type": "ship_type",
        }
    )

    # Rockets (optional features, we won’t use them yet in the model)
    rockets = rockets.rename(
        columns={
            "active": "rocket_active",
            "type": "rocket_type",
        }
    )

    # -------------------------------
    # Merge everything into one big df
    # -------------------------------
    df = launches.merge(launchpads, on="launchpad_id", how="left")
    df = df.merge(cores, on="core_id", how="left")
    df = df.merge(capsules, on="capsule_id", how="left")
    df = df.merge(payloads, on="payload_id", how="left")
    df = df.merge(ships, on="ship_id", how="left")
    df = df.merge(rockets, on="rocket_id", how="left")

    return df
