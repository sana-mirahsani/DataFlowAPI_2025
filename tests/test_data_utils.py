# ==================================================
# This file contains test units for functions in 
# data_utils.py
# ==================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import pytest
import sys
import os

sys.path.append(os.path.abspath("../src"))
from data_utils import load_data, save_data, explore_data, clean_data, create_text_column, clean_text, convert_by_TF_IDF, split_data

# =============================================================================
# 1. Test clean_text
# =============================================================================