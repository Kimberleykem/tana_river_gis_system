import hashlib
import streamlit as st

# Pre-configured system accounts with SHA-256 hashed passwords
USERS_DB = {
    "coordinator@ndma.go.ke": {
        "name": "Sarah Wanjiku (NDMA Coordinator)",
        "role": "Disaster Coordinator",
        "password_hash": hashlib.sha256("ndma2026".encode()).hexdigest(),
        "can_edit_weights": True,
        "can_allocate_kits": True,
    },
    "field.tana@ndma.go.ke": {
        "name": "Hassan Guyo (Field Relief Officer)",
        "role": "Field Relief Officer",
        "password_hash": hashlib.sha256("relief2026".encode()).hexdigest(),
        "can_edit_weights": False,
        "can_allocate_kits": False,
    },
    "analyst@ndma.go.ke": {
        "name": "David Kamau (GIS & ML Analyst)",
        "role": "Data & ML Analyst",
        "password_hash": hashlib.sha256("analyst2026".encode()).hexdigest(),
        "can_edit_weights": True,
        "can_allocate_kits": False,
    },
}


def verify_login(email: str, password: str):
  user = USERS_DB.get(email.strip().lower())
  if not user:
    return None
  entered_hash = hashlib.sha256(password.encode()).hexdigest()
  if entered_hash == user["password_hash"]:
    return user
  return None


def render_login_page():
  st.markdown(
      """
        <div style="background: linear-gradient(90deg, #7f0000 0%, #99000d 100%);
                    color: white; padding: 22px 28px; border-radius: 8px; margin-bottom: 24px;">
            <h2 style="margin:0; font-size:1.6rem;">Tana River County Government & NDMA</h2>
            <p style="margin:6px 0 0 0; opacity:0.9;">Spatial Disaster Relief Prioritization & Decision-Support Portal — Secure Login</p>
        </div>
    """,
      unsafe_allow_html=True,
  )

  col1, col2 = st.columns([1.1, 1])

  with col1:
    st.subheader("Sign In to Dashboard")
    with st.form("login_form"):
      email = st.text_input("Official Email", placeholder="e.g. field.tana@ndma.go.ke")
      password = st.text_input("Password", type="password", placeholder="Enter password")
      submitted = st.form_submit_button("Login", use_container_width=True)

      if submitted:
        user = verify_login(email, password)
        if user:
          st.session_state["authenticated"] = True
          st.session_state["user"] = user
          st.session_state["user_email"] = email.strip().lower()
          st.rerun()
        else:
          st.error("Invalid email or password. Please check your credentials.")

  with col2:
    st.info("""
            **Demo Role Credentials (For System Testing & Defense):**
            * **Disaster Coordinator (Full Control):**  
              `coordinator@ndma.go.ke` / `ndma2026`
            * **Field Relief Officer (Field Reports & Dispatch View):**  
              `field.tana@ndma.go.ke` / `relief2026`
            * **Data & ML Analyst (AHP Calibration & Model Metrics):**  
              `analyst@ndma.go.ke` / `analyst2026`
            """)


def render_user_guide():
  st.header("System User Guide & Operational Manual")
  st.markdown("""
    This platform integrates **Analytic Hierarchy Process (AHP)** multi-criteria weighting with a **Random Forest Machine Learning Classifier** to prioritize humanitarian relief across the **15 wards of Tana River County** (Tana North, Galole, and Tana Delta).
    """)

  with st.expander("1. Understanding User Roles & Permissions", expanded=True):
    st.markdown("""
        * **Disaster Coordinator:** Configures total available relief kits, calibrates AHP vulnerability criteria weights, reviews ML priority classifications, and exports official dispatch manifests.
        * **Field Relief Officer:** Views live ward vulnerability maps and allocation schedules, logs real-time field flood depth observations, and downloads ward dispatch sheets for convoy deployment.
        * **Data & ML Analyst:** Audits model training/validation accuracy, tests sensitivity of AHP criteria weights, and validates spatial overlays.
        """)

  with st.expander("2. How Vulnerability & Priority Scores Are Calculated", expanded=True):
    st.markdown("""
        1. **Socio-Economic & Hazard Inputs:** Each ward is evaluated across 6 indicators sourced from KNBS Census, KIHBS Poverty Reports, and Google/UNOSAT Flood Inundation layers (`Overall Poverty Rate`, `Food Poverty Incidence`, `Under-5 Child Share`, `Elderly 65+ Share`, `Disability Prevalence`, and `Historical Flood Depth`).
        2. **AHP Weighted Scoring:** The sidebar sliders assign relative weights to each indicator to compute a composite vulnerability index between `0.0` and `1.0`.
        3. **Random Forest Classification:** The trained Random Forest model classifies wards into **Critical**, **High**, **Medium**, or **Low** priority tiers, distributing available relief kits proportionally to high-risk populations.
        """)

  with st.expander("3. Step-by-Step Workflow for Relief Allocation", expanded=False):
    st.markdown("""
        * **Step 1 (Log In):** Sign in using your assigned role credentials.
        * **Step 2 (Configure Supplies — Coordinator):** Enter the total relief kits available in the warehouse (e.g., `10,000` units) on the left sidebar.
        * **Step 3 (Inspect Map & Schedule):** Use the **Interactive Dashboard** tab to inspect color-coded ward markers (`Dark Red = Critical`, `Orange = High`, `Yellow/Green = Moderate/Low`) and review exact kit allocations per ward.
        * **Step 4 (Submit Field Update — Field Officer):** Switch to the **Field Flood Report** tab during an active emergency to submit updated water depth readings for any ward.
        * **Step 5 (Export Manifest):** Click **Download Ward Allocation Schedule (CSV)** to generate a printable dispatch report for field teams.
        """)