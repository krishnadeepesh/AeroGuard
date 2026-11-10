"""
AeroGuard - Automated Test Suite
Validates model loading, inference pipelines, RAG retrieval, and edge cases.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import FEATURE_COLUMNS, AQI_CLASSES
from rag_engine import RAGEngine

class TestAeroGuardPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(BASE_DIR, "models")
        cls.rf_model = joblib.load(os.path.join(cls.models_dir, "aqi_random_forest.joblib"))
        cls.baseline_model = joblib.load(os.path.join(cls.models_dir, "aqi_logistic_regression.joblib"))
        cls.scaler = joblib.load(os.path.join(cls.models_dir, "scaler.joblib"))
        cls.rag_engine = RAGEngine(use_ollama=True)

    def test_model_artifacts_exist(self):
        """Verify model files and scalers are serialized on disk."""
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "aqi_random_forest.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "aqi_logistic_regression.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "scaler.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.models_dir, "evaluation_results.json")))

    def test_normal_case_pristine(self):
        """Normal Test Case 1: Pristine clean air reading should classify as Good."""
        sample = pd.DataFrame([{
            "PM2.5": 12.0, "PM10": 25.0, "NO2": 15.0, "SO2": 8.0,
            "CO": 0.4, "O3": 20.0, "Temperature": 26.0, "Humidity": 80.0
        }])[FEATURE_COLUMNS]
        sample_scaled = self.scaler.transform(sample)
        pred = self.rf_model.predict(sample_scaled)[0]
        self.assertEqual(pred, "Good")

    def test_normal_case_severe_smog(self):
        """Normal Test Case 2: Extreme smog concentrations should classify as Severe."""
        sample = pd.DataFrame([{
            "PM2.5": 350.0, "PM10": 550.0, "NO2": 320.0, "SO2": 65.0,
            "CO": 8.0, "O3": 120.0, "Temperature": 9.0, "Humidity": 90.0
        }])[FEATURE_COLUMNS]
        sample_scaled = self.scaler.transform(sample)
        pred = self.rf_model.predict(sample_scaled)[0]
        self.assertEqual(pred, "Severe")

    def test_edge_case_sensor_anomaly(self):
        """Edge Case 1: Borderline threshold boundary between Satisfactory and Moderate."""
        # PM2.5 = 62.0 ug/m3 crosses the CPCB 60.0 boundary into Moderate
        sample = pd.DataFrame([{
            "PM2.5": 62.0, "PM10": 110.0, "NO2": 55.0, "SO2": 20.0,
            "CO": 1.4, "O3": 45.0, "Temperature": 25.0, "Humidity": 60.0
        }])[FEATURE_COLUMNS]
        sample_scaled = self.scaler.transform(sample)
        pred = self.rf_model.predict(sample_scaled)[0]
        # At boundary, model classifies as Moderate or Satisfactory
        self.assertIn(pred, ["Moderate", "Satisfactory"])

    def test_rag_retrieval_grounding(self):
        """Tests that RAG returns non-empty passages with appropriate metadata."""
        docs = self.rag_engine.retrieve("PM2.5 mask N95 respirator advice", top_k=2)
        self.assertGreaterEqual(len(docs), 1)
        self.assertIn("source", docs[0])
        self.assertIn("section", docs[0])
        self.assertIn("text", docs[0])

if __name__ == "__main__":
    unittest.main()
