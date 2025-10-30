# BIRCH-AE: Hierarchical Ensemble Framework for E-Commerce User Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.18+](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-green.svg)](https://ieeexplore.ieee.org)

A scalable hierarchical ensemble clustering framework that combines **BIRCH** (Balanced Iterative Reducing and Clustering using Hierarchies) with **deep autoencoder** feature learning for large-scale e-commerce user segmentation.

## Key Features

- **Hierarchical Ensemble Architecture**: Multiple BIRCH configurations with varying threshold parameters
- **Deep Autoencoder Feature Learning**: Non-linear dimensionality reduction for high-dimensional behavioral features
- **Advanced Consensus Strategies**: Four ensemble methods (MV, WV, AASC, BOHC/CSPA)
- **Dynamic Model Selection**: Automatic strategy selection using multi-criteria evaluation
- **Memory Efficient**: Leverages BIRCH's CF Tree structure for processing millions of users
- **Incremental Learning**: Supports streaming data with real-time segment updates

## Overview

BIRCH-AE addresses three critical challenges in e-commerce user segmentation:

1. **Scalability**: Processes datasets with millions of users through BIRCH's memory-efficient CF Tree structure
2. **High Dimensionality**: Handles correlated behavioral features via deep autoencoder compression
3. **Parameter Sensitivity**: Mitigates BIRCH's threshold sensitivity through ensemble aggregation

### Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    E-Commerce User Data                         │
│            (Behavioral Features: Views, Carts, etc.)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Data Preprocessing & Feature Engineering           │
│    • Numeric: KNN Imputation + StandardScaler                   │
│    • Categorical: Mode Imputation + OneHotEncoder               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Autoencoder Feature Learning (Optional)               │
│    Architecture: 512 → 256 → 128 → Latent(14) → 128 → 256       │
│    • Handles correlated variables                               │
│    • Non-linear dimensionality reduction                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BIRCH Ensemble Generation                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Fine-Grained │  │   Balanced   │  │Coarse-Grained│           │
│  │   T = 0.3    │  │   T = 0.5    │  │   T = 0.8    │           │
│  │   B = 50     │  │   B = 50     │  │   B = 50     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  Multiple cluster counts: K ∈ {5, 10, 15, 20}                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Ensemble Consensus Strategies                      │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌────────────────┐          │
│  │ Majority │ │ Weighted │ │ AASC │ │ BOHC (CSPA)    │          │
│  │ Voting   │ │ Voting   │ │      │ │ Hierarchical   │          │
│  └──────────┘ └──────────┘ └──────┘ └────────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            Dynamic Selection & Evaluation                       │
│  Metrics: Silhouette | Calinski-Harabasz | Davies-Bouldin       │
│  ➜ Automatically selects best ensemble strategy                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Final User Segments                           │
│            (Optimized for personalization)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lisali525/Birch-Autoencoder-Framework.git
cd Birch-Autoencoder-Framework

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from birch_ae import BIRCHAE

# Initialize framework
birch_ae = BIRCHAE(
    latent_dim=14,                    # Autoencoder latent space dimension
    n_clusters_range=[5, 10, 15, 20]  # Cluster counts to evaluate
)

# Fit on your e-commerce data
birch_ae.fit(
    filepath='ecommerce_users.csv',
    user_id_col='visitorid',
    reduction_method='autoencoder',    # or 'pca' for faster processing
    sample_size=50000                  # Optional: for large datasets
)

# Get user segments (best model auto-selected)
segments = birch_ae.get_user_segments('BOHC_10')
print(segments.head())

# Save results
birch_ae.save_results('output/segmentation_results')
```

### Output

```
==============================================================
BIRCH-AE Framework - Starting Pipeline
==============================================================

[1/5] Loading and preprocessing data...
Found 45 numeric and 2 categorical features
Preprocessing complete: Train shape (35000, 78), Test shape (15000, 78)

[2/5] Applying dimensionality reduction: autoencoder...
Training autoencoder on 35000 samples...
Epoch 50/100 - loss: 0.0234 - val_loss: 0.0245
Final data shape: (50000, 14)

[3/5] Running BIRCH ensemble with threshold variations...
✓ BIRCH_Fine-Grained_5: Silhouette=0.523, T=0.3
✓ BIRCH_Balanced_5: Silhouette=0.548, T=0.5
✓ BIRCH_Coarse-Grained_5: Silhouette=0.501, T=0.8
...

[4/5] Running ensemble consensus strategies (MV, WV, AASC, BOHC)...
✓ Majority_Voting_5: Silhouette=0.470
✓ Weighted_Voting_5: Silhouette=0.515
✓ AASC_5: Silhouette=0.548
✓ BOHC_5: Silhouette=0.548
...

Ensemble Improvement: 23.4%
```

## Experimental Results

Performance on large-scale e-commerce datasets:

| Dataset | Users | Features | Best Method | Silhouette | Improvement |
|---------|-------|----------|-------------|------------|-------------|
| Retail Rocket | 1.4M | 45 | BOHC | 0.548 | +23% |
| E-commerce 2019 | 4.1M | 38 | AASC | 0.521 | +19% |

**Key Findings:**
- **23% average improvement** over single BIRCH models
- **Near-linear scalability** to millions of users
- **Robust performance** across multiple ensemble strategies
- **Effective handling** of high-dimensional correlated features

## Architecture Details

### BIRCH Ensemble Configuration

```python
# Three threshold configurations capture different granularities:

Fine-Grained (T=0.3):
  - Many small, homogeneous clusters
  - Captures subtle behavioral differences
  
Balanced (T=0.5):
  - Moderate cluster size and homogeneity
  - Recommended starting point
  
Coarse-Grained (T=0.8):
  - Fewer, larger clusters
  - Broad user categories
```

### Ensemble Consensus Methods

1. **Majority Voting (MV)**
   - Simple democratic voting across ensemble members
   - Fast, interpretable baseline

2. **Weighted Voting (WV)**
   - Quality-weighted aggregation: `w_m = exp(β·S_m) / Σ exp(β·S_j)`
   - Higher quality clusterings have more influence

3. **AASC (Advanced Affinity-based Spectral Clustering)**
   - Builds co-association matrix
   - Applies spectral clustering for consensus

4. **BOHC (BIRCH-Optimized Hierarchical Consensus)**
   - Also known as CSPA (Cluster-based Similarity Partitioning)
   - Preserves hierarchical structure information
   - Best performer in experiments (avg. Silhouette: 0.548)

## 🔧 Configuration

### Key Parameters

```python
BIRCHAE(
    latent_dim=14,                      # Latent space dimension (10-20 recommended)
    n_clusters_range=[5, 10, 15, 20]    # Cluster counts to evaluate
)

.fit(
    filepath='data.csv',                # Path to CSV file
    user_id_col='user_id',              # User ID column name
    reduction_method='autoencoder',     # 'autoencoder' or 'pca'
    sample_size=None,                   # Optional sampling for large datasets
    use_autoencoder=True                # Enable/disable autoencoder
)
```

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB+ RAM, 8+ CPU cores
- **For 1M+ users**: 32GB+ RAM or use sampling

### Performance Tips

- Use `reduction_method='pca'` for faster processing (linear method)
- Set `sample_size` for datasets >100k users
- Enable incremental learning for streaming data
- Parallel processing of BIRCH ensemble members

## Use Cases

### E-Commerce Applications

1. **Customer Segmentation**: Identify distinct user groups for targeted marketing
2. **Personalized Recommendations**: Tailor product suggestions by segment
3. **Churn Prediction**: Segment users by engagement level
4. **Dynamic Pricing**: Optimize pricing strategies per segment
5. **Inventory Management**: Forecast demand by user segment

### Real-World Examples

```python
# Example 1: Retail Rocket Dataset
birch_ae.fit(
    filepath='retail_rocket_users.csv',
    user_id_col='visitorid',
    reduction_method='autoencoder',
    sample_size=100000
)

# Example 2: Real-time Segment Updates
# Process new users incrementally without full re-clustering
new_users = load_new_user_data()
updated_segments = birch_ae.update_segments(new_users)
```

## Citation

If you use BIRCH-AE in your research, please cite:

```bibtex
@article{li2025birchae,
  title={BIRCH-AE: A Scalable Hierarchical Ensemble Framework for E-Commerce User Segmentation with Autoencoder Feature Learning},
  author={Li, Caiwen and others},
  journal={IEEE Access},
  year={2025},
  note={Under Review}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research conducted at Universiti Putra Malaysia (UPM)
- Datasets: [Retail Rocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), [E-commerce Behavior 2019](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- Built on scikit-learn's BIRCH implementation
- Inspired by ensemble clustering research and deep learning advances

## Contact

- **Author**: Caiwen Li
- **Institution**: Universiti Putra Malaysia

## Related Work

- [Extended Regularized K-Means for High-Dimensional Segmentation](https://ieeexplore.ieee.org/document/9893964) (IEEE Access 2022)
- [Cluster-N-Engage: User Engagement Framework](https://ieeexplore.ieee.org/document/10171460) (IEEE Access 2023)
- [BIRCH: A New Data Clustering Algorithm](https://dl.acm.org/doi/10.1145/233269.233324) (SIGMOD 1996)

---

 **Star this repository** if you find it helpful!

 **Read the full paper** for detailed methodology and experiments

 **Report issues** to help improve the framework
