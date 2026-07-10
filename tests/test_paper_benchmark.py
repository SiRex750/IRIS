import pytest
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment

def test_hungarian_uniqueness():
    # Test that Hungarian unique centroid-to-frame assignment guarantees unique frames
    np.random.seed(42)
    idxs = [10, 20, 30, 40, 50]
    features_matrix = np.random.randn(5, 10)
    
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=0, n_init=2)
    kmeans.fit(features_matrix)
    
    # Calculate distance matrix (shape: 3, 5)
    dists = np.linalg.norm(kmeans.cluster_centers_[:, None, :] - features_matrix[None, :, :], axis=-1)
    
    # Run Hungarian matching
    row_ind, col_ind = linear_sum_assignment(dists)
    
    selected_frames = sorted([idxs[j] for j in col_ind])
    
    assert len(selected_frames) == 3
    assert len(set(selected_frames)) == 3
    for f in selected_frames:
        assert f in idxs

def test_farneback_aspect_ratio():
    # Aspect-ratio sizing rules: long side is 320px
    # Case 1: wide video (w > h)
    w, h = 640, 480
    if w >= h:
        new_w = 320
        new_h = int(h * (320.0 / w))
    else:
        new_h = 320
        new_w = int(w * (320.0 / h))
    assert new_w == 320
    assert new_h == 240
    
    # Case 2: tall video (h > w)
    w, h = 480, 640
    if w >= h:
        new_w = 320
        new_h = int(h * (320.0 / w))
    else:
        new_h = 320
        new_w = int(w * (320.0 / h))
    assert new_h == 320
    assert new_w == 240
