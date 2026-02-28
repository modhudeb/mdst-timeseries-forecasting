# MDST Time Series Forecasting with PatchedTalNet

This repository contains four notebook experiments that apply a custom neural architecture (PatchedTalNet) to multivariate time series forecasting.

Forecasting objective used across notebooks:
- Input context window: 168 time steps
- Forecast horizon: 24 time steps
- Multi-output prediction over all variables

## Notebooks Covered

| Notebook | Dataset | Data shape from `df.info()` | Variables | Time resolution note |
|---|---|---|---:|---|
| `mdst_exchange_new.ipynb` | Exchange Rate | 7588 x 8 | 8 | Daily (24 steps = 24 days ) |
| `mdst_electricity.ipynb` | Electricity | 26304 x 321 | 321 | 24 steps marked as 6 hours  |
| `mdst_solar_power.ipynb` | Solar Power | 52560 x 137 | 137 | 24 steps marked as 4 hours  |
| `mdst_traffic.ipynb` | Traffic | 17544 x 862 | 862 | 24 steps marked as 2 hours  |

## Model Architecture Figure

Architecture image:

![PatchedTalNet Architecture](./assets/images/PatchedTalnet_Architecture.png)

## Common Pipeline Used in the Four Notebooks

1. Data loading from compressed `.txt.gz` files.
2. Chronological split with `split_ratio = 0.7`.
3. Normalization with train-only fit (`MinMaxScaler`), then test transform.
4. Sliding window dataset creation:
   - `X`: `[window_size, n_nodes] = [168, n_nodes]`
   - `Y`: `[forecast_horizon, n_nodes] = [24, n_nodes]`
5. TensorFlow input pipeline with batch + prefetch.
6. Training with early stopping based on validation MAE.

## PatchedTalNet: Easy-English Signal Explanation

The same core architecture is used in all four notebooks.

### 1) RevIN (Reversible Instance Normalization)

What it does to input signals:
- For each sample and each variable, computes mean and standard deviation across time.
- Normalizes the input to reduce scale and level shift.
- Saves normalization statistics.
- After prediction, applies inverse transform (denormalization) to return forecasts to the original scale.

Why this helps:
- Many time series are non-stationary (level and volatility drift over time).
- RevIN improves training stability by separating pattern learning from temporary scale drift.

### 2) PatchEmbedding

What it does to input signals:
- Cuts a long sequence into overlapping local windows (patches).
- Flattens each patch and projects it to a fixed embedding dimension (`d_model`).
- Adds learnable positional embedding to preserve order information.

Why this helps:
- Reduces sequence length for attention.
- Preserves local temporal behavior before global modeling.

Concrete example from Exchange notebook:
- Input: 168 steps, 8 variables
- `patch_len=24`, `stride=8`
- Number of patches: `1 + (168 - 24) / 8 = 19`
- Each patch vector before projection: `24 * 8 = 192`
- Token sequence after projection: 19 tokens in `d_model` space

### 3) Transformer Blocks + BiLSTM Pooling

What it does to input signals:
- Transformer self-attention connects each patch with all other patches.
- Residual + normalization + feed-forward blocks refine the representation.
- A BiLSTM pooling layer compresses the patch-token sequence into a compact global summary vector.
- Dense head maps summary vector to `24 x n_nodes` outputs.
- RevIN denormalization maps predictions back to original units.

Why this helps:
- Transformer captures long-range interactions.
- BiLSTM pooling provides strong sequence summarization before regression.

## Per-Notebook Model Configuration

| Dataset Notebook | `patch_len` | `patch_stride` | `d_model` | `n_heads` | `n_stack` | Loss | Optimizer LR |
|---|---:|---:|---:|---:|---:|---|---:|
| Exchange (`mdst_exchange_new.ipynb`) | 24 | 8 | 512 | 32 | 5 | Huber (`delta=1.0`) | `1e-5` |
| Electricity (`mdst_electricity.ipynb`) | 16 | 8 | 64 | 4 | 2 | MSE | `3e-4` |
| Solar (`mdst_solar_power.ipynb`) | 16 | 4 | 128 | 8 | 1 | Huber (`delta=1.0`) | `1e-5` |
| Traffic (`mdst_traffic.ipynb`) | 16 | 8 | 64 | 4 | 2 | Huber (`delta=1.0`) | `3e-4` |

## Accuracy Table (Detailed; From Notebook Outputs)


| Dataset | Step | R2 | RMSE | MAE | MSE | SMAPE | RSE | CORR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exchange | 3  | 0.9858 | 0.0410 | 0.0300 | 0.0017 | 6.22% | 0.1190 | 0.9930 |
| Exchange | 6  | 0.9845 | 0.0429 | 0.0313 | 0.0018 | 6.49% | 0.1245 | 0.9923 |
| Exchange | 12 | 0.9825 | 0.0457 | 0.0334 | 0.0021 | 6.80% | 0.1324 | 0.9913 |
| Exchange | 24 | 0.9779 | 0.0514 | 0.0373 | 0.0026 | 7.53% | 0.1487 | 0.9891 |
| Electricity | 3  | 0.9159 | 0.0567 | 0.0375 | 0.0032 | 13.61% | 0.2900 | 0.9571 |
| Electricity | 6  | 0.9148 | 0.0571 | 0.0377 | 0.0033 | 13.67% | 0.2918 | 0.9565 |
| Electricity | 12 | 0.9135 | 0.0575 | 0.0380 | 0.0033 | 13.73% | 0.2941 | 0.9558 |
| Electricity | 24 | 0.9111 | 0.0583 | 0.0384 | 0.0034 | 13.82% | 0.2982 | 0.9545 |
| Solar | 3  | 0.9217 | 0.0697 | 0.0396 | 0.0049 | 134.93% | 0.2799 | 0.9638 |
| Solar | 6  | 0.9144 | 0.0729 | 0.0415 | 0.0053 | 135.47% | 0.2926 | 0.9601 |
| Solar | 12 | 0.8957 | 0.0805 | 0.0458 | 0.0065 | 136.61% | 0.3230 | 0.9510 |
| Solar | 24 | 0.8531 | 0.0955 | 0.0541 | 0.0091 | 138.64% | 0.3833 | 0.9277 |
| Traffic | 3  | 0.7811 | 0.0264 | 0.0127 | 0.0007 | 30.40% | 0.4679 | 0.8838 |
| Traffic | 6  | 0.7789 | 0.0265 | 0.0127 | 0.0007 | 30.16% | 0.4702 | 0.8826 |
| Traffic | 12 | 0.7754 | 0.0267 | 0.0128 | 0.0007 | 30.02% | 0.4740 | 0.8805 |
| Traffic | 24 | 0.7705 | 0.0270 | 0.0130 | 0.0007 | 30.08% | 0.4791 | 0.8774 |

## Result Table (Step-24 Across the Four Notebooks)

| Dataset | R2 | RMSE | MAE | MSE | SMAPE | RSE | CORR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exchange | 0.9779 | 0.0514 | 0.0373 | 0.0026 | 7.53% | 0.1487 | 0.9891 |
| Electricity | 0.9111 | 0.0583 | 0.0384 | 0.0034 | 13.82% | 0.2982 | 0.9545 |
| Solar | 0.8531 | 0.0955 | 0.0541 | 0.0091 | 138.64% | 0.3833 | 0.9277 |
| Traffic | 0.7705 | 0.0270 | 0.0130 | 0.0007 | 30.08% | 0.4791 | 0.8774 |

## Comparison Table (External Horizon-24 Results on Exchange)

The table below summarizes horizon-24 Exchange results reported in MDST-GNN (Table 3), where metrics are RRMSE and CORR.

| Model | Exchange H=24 RRMSE | Exchange H=24 CORR |
|---|---:|---:|
| AR | 0.1140 | 0.9531 |
| LRidge | 0.0698 | 0.9754 |
| LSVR | 0.0745 | 0.9749 |
| GP | 0.0667 | 0.9785 |
| SETAR | 0.0844 | 0.9725 |
| MLP | 0.1001 | 0.9698 |
| RNN-GRU | 0.0580 | 0.9810 |
| LSTNet | 0.0704 | 0.9751 |
| TPA-LSTM | 0.0593 | 0.9796 |
| MTGNN | 0.0506 | 0.9862 |
| StemGNN | 0.0448 | 0.9917 |
| MDST-GNN | 0.0425 | 0.9934 |


## Authentic Dataset Sources

### Benchmark dataset repository (canonical format used in many papers)
- https://github.com/laiguokun/multivariate-time-series-data
- Exchange: https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate
- Electricity: https://github.com/laiguokun/multivariate-time-series-data/tree/master/electricity
- Solar: https://github.com/laiguokun/multivariate-time-series-data/tree/master/solar-energy
- Traffic: https://github.com/laiguokun/multivariate-time-series-data/tree/master/traffic

### Direct mirror links used in notebook download comments
- Exchange: https://raw.githubusercontent.com/yiminghzc/MDST-GNN/main/MDST-GNN/data/exchange_rate.txt.gz
- Electricity: https://raw.githubusercontent.com/yiminghzc/MDST-GNN/main/MDST-GNN/data/electricity.txt.gz
- Solar: https://raw.githubusercontent.com/yiminghzc/MDST-GNN/main/MDST-GNN/data/solar_AL.txt.gz
- Traffic: https://raw.githubusercontent.com/yiminghzc/MDST-GNN/main/MDST-GNN/data/traffic.txt.gz

### Original raw data sources referenced by benchmark repository
- Electricity (UCI): https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
- Traffic (PeMS): http://pems.dot.ca.gov
- Solar (NREL): http://www.nrel.gov/grid/solar-power-data.html

## Table Reference Map

| Table | Content | Primary source |
|---|---|---|
| Notebooks Covered | Dataset sizes and notebook coverage | `df.info()` outputs in the four notebooks |
| Per-Notebook Model Configuration | Hyperparameters and losses | `model = PatchedTalNet(...)` and `model.compile(...)` cells in notebooks |
| Accuracy Table (Detailed) | Step-wise metrics for each dataset | Printed outputs from `step_all_metrics` evaluation cells in notebooks |
| Comparison Table A | Step-24 cross-dataset summary | Derived from the same notebook metric outputs |
| Comparison Table B | External horizon-24 Exchange benchmarks | MDST-GNN paper, Applied Sciences 2022, Table 3 |

## References

1. LSTNet dataset benchmark repository: https://github.com/laiguokun/multivariate-time-series-data
2. Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks", arXiv:1703.07015, 2017. https://arxiv.org/abs/1703.07015
3. Shih et al., "Temporal Pattern Attention for Multivariate Time Series Forecasting", arXiv:1809.04206, 2018. https://arxiv.org/abs/1809.04206
4. Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks" (MTGNN), arXiv:2005.11650, 2020. https://arxiv.org/abs/2005.11650
5. Cao et al., "StemGNN: Spectral Temporal Graph Neural Network for Multivariate Time-Series Forecasting", arXiv:2005.11946, 2020. https://arxiv.org/abs/2005.11946
6. Hossain et al., "MDST-GNN: A Multi-Dimensional Spatial-Temporal Graph Neural Network for Time-Series Forecasting", Applied Sciences 2022, 12(23), 12064. https://doi.org/10.3390/app122312064
