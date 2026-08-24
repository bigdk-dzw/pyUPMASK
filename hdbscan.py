"""Compatibility shim for the experimental pyUPMASK branch.

The upstream code imports ``hdbscan.HDBSCAN``.  Modern scikit-learn ships a
compatible HDBSCAN estimator, so this shim keeps the existing pyUPMASK
clustering dispatch unchanged while avoiding an extra runtime dependency.
"""
from sklearn.cluster import HDBSCAN

__all__ = ["HDBSCAN"]
