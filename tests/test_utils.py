#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for utility functions.
"""

import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import get_season, format_elapsed_time


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_get_season(self):
        """Test get_season function."""
        # Define season months
        season_months = {
            "winter": ["12", "01", "02"],
            "spring": ["03", "04", "05"],
            "summer": ["06", "07", "08"],
            "autumn": ["09", "10", "11"]
        }
        
        # Test winter
        self.assertEqual(get_season(datetime(2020, 1, 15), season_months), "winter")
        self.assertEqual(get_season(datetime(2020, 12, 25), season_months), "winter")
        
        # Test spring
        self.assertEqual(get_season(datetime(2020, 3, 1), season_months), "spring")
        self.assertEqual(get_season(datetime(2020, 5, 31), season_months), "spring")
        
        # Test summer
        self.assertEqual(get_season(datetime(2020, 6, 15), season_months), "summer")
        self.assertEqual(get_season(datetime(2020, 8, 31), season_months), "summer")
        
        # Test autumn
        self.assertEqual(get_season(datetime(2020, 9, 1), season_months), "autumn")
        self.assertEqual(get_season(datetime(2020, 11, 30), season_months), "autumn")
    
    def test_format_elapsed_time(self):
        """Test format_elapsed_time function."""
        # Test seconds only
        self.assertEqual(format_elapsed_time(45.5), "45.5s")
        
        # Test minutes and seconds
        self.assertEqual(format_elapsed_time(125.3), "2m 5.3s")
        
        # Test hours, minutes, and seconds
        self.assertEqual(format_elapsed_time(3725.7), "1h 2m 5.7s")


if __name__ == "__main__":
    unittest.main()