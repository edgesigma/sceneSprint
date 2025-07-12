"""
Unit and Integration Tests for Context-Aware Feature Extraction
==============================================================
"""

import os
import unittest
import numpy as np
from PIL import Image
import sys
import shutil

# Add parent directory to path to import the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_aware_feature_extraction import (
    calculate_color_histogram,
    bin_person_count,
    extract_features,
    process_images_in_directory,
    H_BINS, S_BINS, V_BINS
)

class TestFeatureExtraction(unittest.TestCase):
    """Test suite for feature extraction functions"""

    def setUp(self):
        """Set up test assets"""
        self.test_dir = 'test_images'
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a solid red image
        self.red_image_path = os.path.join(self.test_dir, 'red.png')
        red_img = Image.new('RGB', (100, 100), color = 'red')
        red_img.save(self.red_image_path)

        # Create a solid blue image
        self.blue_image_path = os.path.join(self.test_dir, 'blue.png')
        blue_img = Image.new('RGB', (100, 100), color = 'blue')
        blue_img.save(self.blue_image_path)

    def tearDown(self):
        """Clean up test assets"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # ------------------
    #  Unit Tests
    # ------------------

    def test_calculate_color_histogram_hsv_4x4(self):
        """Test the 4x4 HSV color histogram calculation"""
        red_img = Image.open(self.red_image_path)
        # The function expects a BGR numpy array, so we convert
        red_img_bgr = cv2.cvtColor(np.array(red_img), cv2.COLOR_RGB2BGR)
        hist = calculate_color_histogram(red_img_bgr, grid_size=(4, 4))
        
        # Total bins = 4*4 * (H_BINS * S_BINS * V_BINS)
        expected_len = 16 * (H_BINS * S_BINS * V_BINS)
        self.assertEqual(len(hist), expected_len)
        
        # Check that values are normalized between 0 and 1
        self.assertTrue(np.all(hist >= 0.0))
        self.assertTrue(np.all(hist <= 1.0))

        # For a solid red image, the max value of the histogram should be 1.0
        # because of normalization.
        self.assertAlmostEqual(np.max(hist), 1.0, places=5)


    def test_bin_person_count(self):
        """Test the person count binning logic"""
        self.assertTrue(np.array_equal(bin_person_count(0), [1, 0, 0, 0]))
        self.assertTrue(np.array_equal(bin_person_count(1), [0, 1, 0, 0]))
        self.assertTrue(np.array_equal(bin_person_count(2), [0, 0, 1, 0]))
        self.assertTrue(np.array_equal(bin_person_count(3), [0, 0, 0, 1]))
        self.assertTrue(np.array_equal(bin_person_count(5), [0, 0, 0, 1]))
        self.assertTrue(np.array_equal(bin_person_count(10), [0, 0, 0, 1]))

    # ------------------
    #  Integration Test
    # ------------------

    def test_process_images_in_directory_integration(self):
        """Integration test for the full feature extraction process"""
        output_file = os.path.join(self.test_dir, 'features.tsv')
        
        # Mock detect_person_count to avoid dependency on mediapipe for this test
        # by using the person_count_override in the extract_features function
        original_extract_features = extract_features
        
        def mock_extract_features(image_path):
            # The real extract_features calls other functions, we can call it
            # with an override to avoid the person detection model
            return original_extract_features(image_path, person_count_override=0)

        # This is a bit tricky, let's just test the main loop
        # We will rely on the person_count_override in the main script's function
        # For a true integration test, we'd need to patch detect_person_count
        
        process_images_in_directory(self.test_dir, output_file, person_count_override=0)
        
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2) # red.png and blue.png
        
        # Check the line for the red image
        red_line = [line for line in lines if 'red.png' in line][0]
        parts = red_line.strip().split('\t')
        self.assertEqual(len(parts), 2)
        
        feature_vector = np.fromstring(parts[1], sep=',')
        
        # Expected length: (16 * H * S * V) + 4 (person bins)
        expected_len = 16 * H_BINS * S_BINS * V_BINS + 4
        self.assertEqual(len(feature_vector), expected_len)
        
        # Person count bin for 0 people should be [1, 0, 0, 0]
        person_bins = feature_vector[-4:]
        np.testing.assert_array_equal(person_bins, [1, 0, 0, 0])


if __name__ == '__main__':
    # Add cv2 to globals for the test to run
    import cv2
    unittest.main()
