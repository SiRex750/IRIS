"""
L1 Elysium — active context cache for IRIS.

Port of HADES cache.py with codec-native pressure signalling.
Residual energy from Charon-V replaces psutil RAM% as the
budget scaling signal. PEAK/SALIENT/CANDIDATE/SKIP tiers
map to 100%/85%/70%/30% budget multipliers respectively.

Eviction key: (1 - pagerank_score) * (1 - source_residual)
Lowest composite score evicted first.

Owner: Track A
"""
from cache import L1Cache

