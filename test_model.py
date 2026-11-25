import unittest
import joblib
import pandas as pd

class TestDisasterPriorityModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the serialized Random Forest model
        cls.model = joblib.load('disaster_priority_rf_model.pkl')
        cls.valid_classes = {'High Priority', 'Medium Priority', 'Low Priority'}

    def test_high_risk_boundary_vector(self):
        # Boundary input: extreme poverty, high vulnerability, severe flood hazard
        sample_input = pd.DataFrame([{
            'poverty_rate': 0.85,
            'elderly_child_ratio': 0.55,
            'disability_rate': 0.12,
            'infrastructure_deficit': 0.90,
            'flood_hazard_level': 5
        }])
        prediction = self.model.predict(sample_input)[0]
        self.assertIn(prediction, self.valid_classes)
        self.assertEqual(prediction, 'High Priority')

    def test_low_risk_boundary_vector(self):
        # Boundary input: low poverty, low vulnerability, minimal flood hazard
        sample_input = pd.DataFrame([{
            'poverty_rate': 0.15,
            'elderly_child_ratio': 0.20,
            'disability_rate': 0.02,
            'infrastructure_deficit': 0.15,
            'flood_hazard_level': 1
        }])
        prediction = self.model.predict(sample_input)[0]
        self.assertIn(prediction, self.valid_classes)
        self.assertEqual(prediction, 'Low Priority')

if __name__ == '__main__':
    unittest.main()