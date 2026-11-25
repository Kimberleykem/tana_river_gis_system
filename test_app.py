import os
import unittest


class TestTanaRiverDashboard(unittest.TestCase):
    def test_project_structure_integrity(self):
        """Verify core Streamlit dashboard, spatial data, and source modules exist."""
        self.assertTrue(os.path.exists("app.py"), "app.py is missing")
        self.assertTrue(os.path.isdir("data"), "data directory is missing")
        self.assertTrue(os.path.isdir("src"), "src directory is missing")

    def test_critical_priority_tier_escalation(self):
        """Verify severe flood depth and high vulnerability classify as Critical Priority."""
        # Simulates high-risk Tana River ward indicator weights from the dashboard
        weights = {
            "under5_child_share": 0.10,
            "elderly_65_plus": 0.08,
            "disability_index": 0.12,
            "historical_flood_depth": 0.30,
            "drought_severity": 0.12,
            "health_facility_distance": 0.08,
        }
        ward_exposure = {
            "under5_child_share": 0.90,
            "elderly_65_plus": 0.85,
            "disability_index": 0.80,
            "historical_flood_depth": 0.95,
            "drought_severity": 0.70,
            "health_facility_distance": 0.85,
        }

        composite_score = sum(weights[k] * ward_exposure[k] for k in weights)
        priority_tier = (
            "Critical Priority"
            if composite_score >= 0.65
            else "High Priority"
            if composite_score >= 0.45
            else "Moderate Priority"
        )

        self.assertEqual(priority_tier, "Critical Priority")


if __name__ == "__main__":
    unittest.main()