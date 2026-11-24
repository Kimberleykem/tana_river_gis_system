import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.auth import render_login_page, render_user_guide
from src.gis_mapper import GISVisualizerAndAllocator
from src.ingestion import DataIngestionManager
from src.ml_engine import PrioritizationMLModel
from src.preprocessor import DEFAULT_AHP_WEIGHTS, VulnerabilityPreprocessor

st.set_page_config(
    page_title="Tana River NDMA Relief Prioritization", layout="wide"
)

# 1. Authentication Gate
if not st.session_state.get("authenticated", False):
  render_login_page()
  st.stop()

current_user = st.session_state["user"]

# 2. Custom Styling & Header Banner
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    .header-banner {
        background: linear-gradient(90deg, #7f0000 0%, #99000d 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .header-banner h2 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
    }
    .header-banner p {
        margin: 4px 0 0 0;
        font-size: 0.92rem;
        color: #f8d7da;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #99000d;
        padding: 12px 16px;
        border-radius: 6px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

top_col1, top_col2 = st.columns([4.5, 1])
with top_col1:
  st.markdown(
      f"""
        <div class="header-banner">
            <h2>Tana River County Government & NDMA | Spatial Disaster Relief Prioritization</h2>
            <p>Ward-Level Multi-Criteria Vulnerability Mapping & Resource Allocation Decision-Support System (15 Wards)  | 
            Active User: <strong>{current_user['name']}</strong></p>
        </div>
    """,
      unsafe_allow_html=True,
  )
with top_col2:
  st.write("")
  if st.button("Logout", use_container_width=True):
    st.session_state.clear()
    st.rerun()

# 3. Role-Aware Sidebar Controls
FEATURE_LABELS = {
    "Poverty_Rate": "Overall Poverty Rate (%)",
    "Food_Poverty": "Food Poverty Incidence (%)",
    "Under5_Pct": "Under-5 Child Share (%)",
    "Elderly_Pct": "Elderly Population 65+ (%)",
    "Disability_Idx": "Disability Prevalence Index",
    "Flood_Inundation_m": "Historical Flood Depth (m)",
    "Drought_Severity": "Drought Severity Index",
    "Dist_Health_km": "Distance to Health Facility (km)",
}

st.sidebar.header("Relief Supply Configuration")
total_kits = st.sidebar.number_input(
    "Available Relief Kits (Units)",
    min_value=500,
    max_value=100000,
    value=10000,
    step=500,
    disabled=not current_user["can_allocate_kits"],
    help=(
        "Only NDMA Disaster Coordinators can adjust total relief kit inventory."
    ),
)

st.sidebar.header("AHP Vulnerability Criteria")
if not current_user["can_edit_weights"]:
  st.sidebar.caption("🔒 Read-only view for Field Relief Officers")

custom_weights = {}
for feat, default_w in DEFAULT_AHP_WEIGHTS.items():
  custom_weights[feat] = st.sidebar.slider(
      FEATURE_LABELS[feat],
      min_value=0.0,
      max_value=0.50,
      value=default_w,
      step=0.02,
      disabled=not current_user["can_edit_weights"],
  )

# 4. Data Ingestion, Live Field Overrides, AHP Preprocessing & ML Inference
ingestor = DataIngestionManager(
    age_csv="data/table_24a_rural_population_by_age_sex_and_county_full.csv",
    dis_csv="data/volume_4_table_227_pwith_disability_by_type_of_disability_sex_area_of_residence_county_and_sub.csv",
    flood_zip="data/google-flood-risk.zip",
)
raw_wards = ingestor.get_integrated_ward_data()
flood_gdf = ingestor.load_flood_shapefile()

# Apply any real-time field officer flood updates stored in session_state
if "field_overrides" in st.session_state:
  for ward_name, updated_depth in st.session_state["field_overrides"].items():
    raw_wards.loc[raw_wards["Ward"] == ward_name, "Flood_Inundation_m"] = (
        updated_depth
    )

preprocessor = VulnerabilityPreprocessor(weights=custom_weights)
scored_wards = preprocessor.fit_transform(raw_wards)

ml_engine = PrioritizationMLModel()
metrics = ml_engine.train_and_evaluate(scored_wards, preprocessor)
scored_wards["ML_Predicted_Tier"] = ml_engine.predict_wards(scored_wards)

gis_alloc = GISVisualizerAndAllocator()
final_plan = gis_alloc.allocate_relief_kits(scored_wards, total_kits=total_kits)

# 5. Main Navigation Tabs
tab_dash, tab_field, tab_guide = st.tabs([
    "🗺️ Interactive GIS Dashboard",
    "📡 Field Officer Flood Report",
    "📘 How to Use (User Guide)",
])

with tab_dash:
  c1, c2, c3, c4 = st.columns(4)
  c1.metric(
      "Critical Priority Wards",
      int((final_plan["Priority_Tier"] == "Critical").sum()),
  )
  c2.metric("Model Training Accuracy", f"{metrics['train_accuracy']}%")
  c3.metric("Model Validation Accuracy", f"{metrics['test_accuracy']}%")
  c4.metric(
      "Total Relief Kits Dispatched", f"{final_plan['Allocated_Kits'].sum():,}"
  )

  st.write("")
  col_map, col_table = st.columns([1.25, 1])

  with col_map:
    st.subheader("Ward Vulnerability & Flood Inundation Map")
    priority_map = gis_alloc.build_map(final_plan, flood_gdf)
    st_folium(priority_map, width=720, height=500)

  with col_table:
    st.subheader("Ward Relief Allocation Schedule")
    table_df = final_plan[[
        "Ward",
        "Sub_County",
        "Pop",
        "Vulnerability_Score",
        "Priority_Tier",
        "ML_Predicted_Tier",
        "Allocated_Kits",
    ]].rename(
        columns={
            "Sub_County": "Sub-County",
            "Pop": "Population",
            "Vulnerability_Score": "AHP Score",
            "Priority_Tier": "AHP Tier",
            "ML_Predicted_Tier": "RF Tier",
            "Allocated_Kits": "Relief Kits",
        }
    )
    st.dataframe(table_df, use_container_width=True, height=430, hide_index=True)

    csv_export = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Dispatch Schedule (CSV)",
        data=csv_export,
        file_name="tana_river_relief_allocation.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_field:
  st.subheader("Field Officer Live Flood Assessment Submission")
  st.write(
      "Log on-the-ground flood depth observations for any ward in Tana River"
      " County to immediately recalculate AHP vulnerability scores, Random"
      " Forest priority tiers, and relief kit allocations."
  )

  with st.form("field_update_form"):
    ward_list = sorted(raw_wards["Ward"].unique().tolist())
    selected_ward = st.selectbox("Select Administrative Ward", ward_list)

    current_val = float(
        raw_wards.loc[
            raw_wards["Ward"] == selected_ward, "Flood_Inundation_m"
        ].values[0]
    )
    observed_depth = st.slider(
        "Observed Flood Inundation Depth (meters)",
        min_value=0.0,
        max_value=5.0,
        value=min(current_val, 5.0),
        step=0.1,
    )
    field_notes = st.text_area(
        "Field Situation Report (e.g., displaced households, submerged"
        " bridges/roads)"
    )

    col_sub, col_reset = st.columns([2, 1])
    with col_sub:
      submit_field = st.form_submit_button(
          "Submit Flood Update & Recalculate Priority", use_container_width=True
      )
    with col_reset:
      reset_field = st.form_submit_button(
          "Reset All Field Overrides", use_container_width=True
      )

    if submit_field:
      if "field_overrides" not in st.session_state:
        st.session_state["field_overrides"] = {}
      st.session_state["field_overrides"][selected_ward] = observed_depth
      if "field_logs" not in st.session_state:
        st.session_state["field_logs"] = []
      st.session_state["field_logs"].append({
          "Submitted By": current_user["name"],
          "Ward": selected_ward,
          "Observed Flood Depth (m)": observed_depth,
          "Situation Notes": field_notes if field_notes else "Routine gauge update",
      })
      st.success(
          f"Updated {selected_ward} Ward flood depth to {observed_depth}m."
          " Switch back to the Interactive GIS Dashboard tab to inspect the"
          " updated map and relief allocations."
      )

    if reset_field:
      st.session_state.pop("field_overrides", None)
      st.session_state.pop("field_logs", None)
      st.info("All field overrides cleared. Baseline satellite flood depths restored.")

  if st.session_state.get("field_logs"):
    st.markdown("#### Recent Field Assessment Log")
    st.dataframe(
        pd.DataFrame(st.session_state["field_logs"]),
        use_container_width=True,
        hide_index=True,
    )

with tab_guide:
  render_user_guide()