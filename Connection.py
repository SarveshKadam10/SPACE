import pandas as pd

def get_data():
    # Load all CSVs
    launches = pd.read_csv("launches.csv")
    cores = pd.read_csv("cores.csv")
    capsules = pd.read_csv("capsules.csv")
    payloads = pd.read_csv("payloads.csv")
    ships = pd.read_csv("ships.csv")
    launchpads = pd.read_csv("launchpads.csv")
    rockets = pd.read_csv("rockets.csv")  # optional

    # Merge CSVs exactly like SQL joins
    df = launches.merge(launchpads, on="launchpad_id", how="left")
    df = df.merge(cores, on="core_id", how="left")
    df = df.merge(capsules, on="capsule_id", how="left")
    df = df.merge(payloads, on="payload_id", how="left")
    df = df.merge(ships, on="ship_id", how="left")

    # Merge rockets only if rocket_id exists
    if "rocket_id" in launches.columns and "rocket_id" in rockets.columns:
        df = df.merge(rockets, on="rocket_id", how="left")

    return df
