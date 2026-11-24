import os
import zipfile
import pandas as pd
import geopandas as gpd


class DataIngestionManager:
    """Handles ingestion and structural auditing of KNBS CSVs and Flood Shapefiles."""

    def __init__(self, age_csv: str, dis_csv: str, flood_zip: str):
        self.age_csv = age_csv
        self.dis_csv = dis_csv
        self.flood_zip = flood_zip

    def load_knbs_demographics(self) -> dict:
        df_age = pd.read_csv(self.age_csv)
        tr_age = df_age[df_age["COUNTY"].str.upper() == "TANARIVER"].copy()
        for col in ["Male", "Female", "Total"]:
            tr_age[col] = pd.to_numeric(tr_age[col], errors="coerce")

        total_pop = float(tr_age.loc[tr_age["Age"] == "Total", "Total"].values[0])
        under_5 = float(tr_age.loc[tr_age["Age"] == "0 - 4", "Total"].values[0])
        elderly_cols = ["65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100+"]
        elderly_65p = float(tr_age[tr_age["Age"].isin(elderly_cols)]["Total"].sum())

        return {
            "total_rural_pop": int(total_pop),
            "under_5_pct": round((under_5 / total_pop) * 100, 2),
            "elderly_65p_pct": round((elderly_65p / total_pop) * 100, 2),
        }

    def audit_and_fix_disability_shift(self) -> pd.DataFrame:
        df_dis_raw = pd.read_csv(self.dis_csv)
        for col in df_dis_raw.columns[1:]:
            df_dis_raw[col] = pd.to_numeric(df_dis_raw[col], errors="coerce")

        tr_raw = df_dis_raw.iloc[26:30].copy()
        cleaned = pd.DataFrame({
            "Area": ["Tana River (County)", "Tana Delta", "Tana North", "Galole"],
            "Visual_Total": tr_raw["Total "].values,
            "Hearing_Total": tr_raw["Visual Total"].values,
            "Mobility_Total": tr_raw["Hearing Total"].values,
            "Cognition_Total": tr_raw["Mobility Total"].values,
            "SelfCare_Total": tr_raw["Cognition Total"].values,
            "Comm_Total": tr_raw["Self-care Total"].values,
        })
        domain_cols = ["Visual_Total", "Hearing_Total", "Mobility_Total", "Cognition_Total", "SelfCare_Total", "Comm_Total"]
        cleaned["Total_Instances"] = cleaned[domain_cols].sum(axis=1)
        return cleaned

    def load_flood_shapefile(self, extract_dir: str = "data/extracted_flood") -> gpd.GeoDataFrame:
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(self.flood_zip, "r") as z:
                z.extractall(extract_dir)
        shp_path = os.path.join(extract_dir, "Google Flood Risk", "google_flood_risk.shp")
        return gpd.read_file(shp_path)

    def get_integrated_ward_data(self) -> pd.DataFrame:
        wards_data = [
            {"Ward": "Garsen Central", "Sub_County": "Tana Delta", "Lat": -2.274, "Lon": 40.112, "Pop": 24100, "Poverty_Rate": 68.5, "Food_Poverty": 48.2, "Under5_Pct": 17.8, "Elderly_Pct": 3.4, "Disability_Idx": 0.88, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.65, "Dist_Health_km": 4.2},
            {"Ward": "Garsen South",   "Sub_County": "Tana Delta", "Lat": -2.415, "Lon": 40.185, "Pop": 21800, "Poverty_Rate": 71.2, "Food_Poverty": 50.1, "Under5_Pct": 18.1, "Elderly_Pct": 3.6, "Disability_Idx": 0.92, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.60, "Dist_Health_km": 9.8},
            {"Ward": "Garsen West",    "Sub_County": "Tana Delta", "Lat": -2.250, "Lon": 39.920, "Pop": 18400, "Poverty_Rate": 72.4, "Food_Poverty": 51.0, "Under5_Pct": 17.9, "Elderly_Pct": 3.5, "Disability_Idx": 0.85, "Flood_Inundation_m": 1.0, "Drought_Severity": 0.78, "Dist_Health_km": 14.5},
            {"Ward": "Garsen North",   "Sub_County": "Tana Delta", "Lat": -2.060, "Lon": 40.080, "Pop": 22300, "Poverty_Rate": 69.0, "Food_Poverty": 47.9, "Under5_Pct": 17.6, "Elderly_Pct": 3.3, "Disability_Idx": 0.89, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.68, "Dist_Health_km": 8.1},
            {"Ward": "Kipini East",    "Sub_County": "Tana Delta", "Lat": -2.510, "Lon": 40.480, "Pop": 16900, "Poverty_Rate": 66.8, "Food_Poverty": 45.5, "Under5_Pct": 17.2, "Elderly_Pct": 3.7, "Disability_Idx": 0.81, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.55, "Dist_Health_km": 12.3},
            {"Ward": "Kipini West",    "Sub_County": "Tana Delta", "Lat": -2.480, "Lon": 40.310, "Pop": 19200, "Poverty_Rate": 70.1, "Food_Poverty": 48.8, "Under5_Pct": 17.7, "Elderly_Pct": 3.5, "Disability_Idx": 0.86, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.58, "Dist_Health_km": 11.0},
            {"Ward": "Kinakomba",      "Sub_County": "Galole",     "Lat": -1.780, "Lon": 40.040, "Pop": 17600, "Poverty_Rate": 67.4, "Food_Poverty": 46.9, "Under5_Pct": 17.4, "Elderly_Pct": 3.2, "Disability_Idx": 0.74, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.72, "Dist_Health_km": 10.4},
            {"Ward": "Mikinduni",      "Sub_County": "Galole",     "Lat": -1.610, "Lon": 40.010, "Pop": 16200, "Poverty_Rate": 66.9, "Food_Poverty": 46.3, "Under5_Pct": 17.5, "Elderly_Pct": 3.1, "Disability_Idx": 0.71, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.74, "Dist_Health_km": 9.2},
            {"Ward": "Chewani",        "Sub_County": "Galole",     "Lat": -1.505, "Lon": 40.025, "Pop": 23800, "Poverty_Rate": 61.5, "Food_Poverty": 43.1, "Under5_Pct": 16.8, "Elderly_Pct": 3.0, "Disability_Idx": 0.76, "Flood_Inundation_m": 1.0, "Drought_Severity": 0.65, "Dist_Health_km": 3.5},
            {"Ward": "Wayu",           "Sub_County": "Galole",     "Lat": -1.550, "Lon": 39.650, "Pop": 19500, "Poverty_Rate": 73.8, "Food_Poverty": 52.4, "Under5_Pct": 18.0, "Elderly_Pct": 3.5, "Disability_Idx": 0.79, "Flood_Inundation_m": 0.5, "Drought_Severity": 0.91, "Dist_Health_km": 19.6},
            {"Ward": "Chewele",        "Sub_County": "Tana North", "Lat": -1.120, "Lon": 39.940, "Pop": 21400, "Poverty_Rate": 65.2, "Food_Poverty": 45.0, "Under5_Pct": 17.3, "Elderly_Pct": 2.9, "Disability_Idx": 0.68, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.79, "Dist_Health_km": 5.8},
            {"Ward": "Hirimani",       "Sub_County": "Tana North", "Lat": -1.040, "Lon": 39.880, "Pop": 18900, "Poverty_Rate": 67.8, "Food_Poverty": 47.1, "Under5_Pct": 17.6, "Elderly_Pct": 3.0, "Disability_Idx": 0.66, "Flood_Inundation_m": 1.0, "Drought_Severity": 0.84, "Dist_Health_km": 11.7},
            {"Ward": "Bangale",        "Sub_County": "Tana North", "Lat": -0.780, "Lon": 39.320, "Pop": 22700, "Poverty_Rate": 74.5, "Food_Poverty": 53.2, "Under5_Pct": 18.2, "Elderly_Pct": 3.4, "Disability_Idx": 0.72, "Flood_Inundation_m": 0.5, "Drought_Severity": 0.95, "Dist_Health_km": 22.4},
            {"Ward": "Sala",           "Sub_County": "Tana North", "Lat": -0.620, "Lon": 39.750, "Pop": 15400, "Poverty_Rate": 71.9, "Food_Poverty": 50.6, "Under5_Pct": 17.9, "Elderly_Pct": 3.2, "Disability_Idx": 0.64, "Flood_Inundation_m": 1.0, "Drought_Severity": 0.89, "Dist_Health_km": 17.9},
            {"Ward": "Madogo",         "Sub_County": "Tana North", "Lat": -0.465, "Lon": 39.635, "Pop": 26500, "Poverty_Rate": 64.0, "Food_Poverty": 44.2, "Under5_Pct": 17.1, "Elderly_Pct": 2.8, "Disability_Idx": 0.70, "Flood_Inundation_m": 5.0, "Drought_Severity": 0.82, "Dist_Health_km": 4.9},
        ]
        return pd.DataFrame(wards_data)