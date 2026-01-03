import pandas as pd
import numpy as np
import pytest
from amenity_complexity.core import count_matrix, rca, specialization, complexity, compute_complexity

def test_rca_toy_matrix():
    # L1 has everything (Diversified)
    # A1 is everywhere (Ubiquitous)
    # Using 'count_matrix' behavior simulation or direct DF
    data = {
        'A1': [10, 10, 10],
        'A2': [10, 0, 0],
        'A3': [10, 5, 0]
    }
    counts = pd.DataFrame(data, index=['L1', 'L2', 'L3'])
    
    # Test RCA calculation
    rca_mat = rca(counts)
    assert rca_mat.shape == counts.shape
    
    # A2 is only in L1. 
    # Global sum = 55. A2 sum = 10. L1 sum = 30.
    # RCA(L1, A2) = (10/30) / (10/55) = (1/3) / (2/11) = 11/6 = 1.8333...
    assert np.isclose(rca_mat.loc['L1', 'A2'], 11/6)
    
    # Test Specialization (Binary)
    M = specialization(rca_mat, threshold=1.0)
    assert M.loc['L1', 'A2'] == 1
    assert M.loc['L2', 'A2'] == 0

def test_complexity_hidalgo():
    # Simple triangular structure (nestedness)
    data = {
        'A1': [1, 1, 1],
        'A2': [1, 0, 0],
        'A3': [1, 1, 0]
    }
    M = pd.DataFrame(data, index=['L1', 'L2', 'L3'])
    
    # Hidalgo method (ECI/PCI style)
    # We expect L1 (most diversified, has rare things) to have high complexity
    # We expect A2 (rarest, found in diversified loc) to have high complexity
    loc_comp, act_comp = complexity(M, method='hidalgo', orient='diversity', zscore=True)
    
    assert len(loc_comp) == 3
    assert len(act_comp) == 3
    
    # Check ordering: L1 > L3 > L2
    assert loc_comp['L1'] > loc_comp['L2']
    
def test_compute_complexity_pipeline():
    # Test the end-to-end function with dummy data
    # Create a long-form DataFrame
    df = pd.DataFrame({
        'h3': ['h1', 'h1', 'h2', 'h3'],
        'category': ['c1', 'c2', 'c1', 'c1']
    })
    
    # Min totals = 1 to keep everything
    profile = compute_complexity(
        df, 
        unit_col='h3', 
        category_col='category', 
        min_unit_total=1, 
        min_category_total=1
    )
    
    assert profile.counts is not None
    assert profile.units is not None
    assert 'complexity_hidalgo' in profile.units.columns
    assert 'complexity_juhasz' in profile.units.columns
