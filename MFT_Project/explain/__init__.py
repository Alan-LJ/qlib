"""Explainability utilities (e.g. SHAP integration) for the MFT project."""

from .shap_explainer import FusionShapExplainer, ShapResult

__all__ = ["FusionShapExplainer", "ShapResult"]
