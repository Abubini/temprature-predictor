"""
Evaluation module for regression models.
Provides tools to load trained models, run evaluations,
generate metrics, and create comparative visualizations.
"""

from .evaluate import (
    load_models_and_data,
    evaluate_model,
    create_regression_visualizations,
    compare_models,
    evaluate_all_models,
)
