"""CUPID performance-influence data curation (Agia et al., 2025).

Score each training demonstration by its estimated influence on closed-loop
success, using station rollouts of a base policy trained on all demonstrations,
then keep the highest-scoring demonstrations for retraining.
"""
