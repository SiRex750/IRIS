import pytest
import numpy as np
from iris.charon_v import compute_motion_geometry

def test_compute_motion_geometry_empty():
    res = compute_motion_geometry([], 320, 240)
    assert res["divergence"] == 0.0
    assert res["curl"] == 0.0
    assert res["jacobian_frobenius"] == 0.0
    assert res["hessian_max_eigenvalue"] == 0.0
    assert res["motion_entropy"] == 0.0

def test_compute_motion_geometry_expansion():
    # Width = 160, Height = 160 -> grid is 10 x 10
    # Create motion vectors directed outward from center (80, 80)
    # gx range: 0 to 9, gy range: 0 to 9. Center is 4.5, 4.5
    motion_vectors = []
    for gy in range(10):
        for gx in range(10):
            # Calculate coordinates at block center
            dst_x = gx * 16 + 8
            dst_y = gy * 16 + 8
            # Outward motion relative to center
            motion_x = 2.0 if gx >= 5 else -2.0
            motion_y = 2.0 if gy >= 5 else -2.0
            motion_vectors.append((dst_x, dst_y, dst_x, dst_y, motion_x, motion_y))
            
    res = compute_motion_geometry(motion_vectors, 160, 160)
    # Divergence should be positive for expansion
    assert res["divergence"] > 0.0
    # No rotation
    assert res["curl"] == 0.0

def test_compute_motion_geometry_rotation():
    # Rotate around center
    motion_vectors = []
    for gy in range(10):
        for gx in range(10):
            dst_x = gx * 16 + 8
            dst_y = gy * 16 + 8
            # Rotational field: motion_x = -(gy - 4.5), motion_y = (gx - 4.5)
            motion_x = -(gy - 4.5)
            motion_y = (gx - 4.5)
            motion_vectors.append((dst_x, dst_y, dst_x, dst_y, motion_x, motion_y))
            
    res = compute_motion_geometry(motion_vectors, 160, 160)
    # Curl should be positive (non-zero)
    assert res["curl"] > 0.0
    # Divergence should be near zero (incompressible rotation)
    assert abs(res["divergence"]) < 1e-5
