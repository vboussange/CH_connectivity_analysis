from pathlib import Path
import sys
import pandas as pd
sys.path.append(str(Path(__file__).parent / Path("../src/")))
from group_preprocessing import compile_group_suitability, GROUP_INFO

if __name__ == "__main__":
    config = {"hash": "277b08f",
              "resolution": 25,}
    base_path = Path(__file__).parent / Path("../../data/processed")  / config["hash"]
    df = pd.DataFrame(columns=["group", "habitat", "species number", "distance", "median mean dispersal range (km)", "std mean dispersal range (km)"])
    for hab in ["Aqu", "Ter"]:
        for group in GROUP_INFO.keys():
            try:
                suitability_dataset = compile_group_suitability(group, hab, config["resolution"])
                D_m = suitability_dataset.attrs["D_m"]
                species_number = len(eval(suitability_dataset.attrs["species"]))
                df = pd.concat([df, pd.DataFrame({"group": [group], 
                                                  "habitat": [hab], 
                                                  "species number": [species_number], 
                                                  "distance": [GROUP_INFO[group]], 
                                                  "median mean dispersal range (km)": [round(D_m / 1000, 2)],
                                                  "std mean dispersal range (km)": [round(suitability_dataset.attrs["D_m_std"] / 1000, 2)]})],
                                                ignore_index=True)
            except Exception as e:
                print(f"Error processing group {group} in habitat {hab}: {e}")
                continue
    df.to_csv(base_path / "metadata.csv", index=False)