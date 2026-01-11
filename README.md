## CDCR-Rank: A Computational Model for Predicting Drug Combination Dose Response using Ranking-Based Optimization


## 📋 Overview

CDCR-Rank is a deep learning framework for predicting drug synergy scores from drug combinations, doses, 
and cell line features. The model incorporates **uRank loss** for learning dose-response relationships and includes 
**integrated gradients-based interpretability** to understand which modalities (drug chemical features, doses, or cell context) drive synergy predictions.

## 📊 Dataset

The model is trained on **NCI-ALMANAC** data, which includes:
- Drug combinations with SMILES representations
- Multiple dose concentrations per combination
- Cell line genomic features
- Measured synergy scores (PercentageGrowth)


## Dependencies
- numpy>=1.19.0
- tensorflow>=2.6.0
- pandas>=1.3.0
- scikit-learn>=0.24.0
- scipy>=1.7.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- tqdm>=4.62.0
