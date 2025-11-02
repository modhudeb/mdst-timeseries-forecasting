# MDST Time Series Forecasting Project

This repository contains implementation of advanced time series forecasting models applied to multiple datasets using TensorFlow and Keras. The project features a novel architecture called **TalNetV2**, a custom-built enhanced version of TalNet with significant improvements, alongside a Stacked Bidirectional LSTM implementation. TalNetV2 represents an original contribution to multivariate time series prediction, incorporating advanced features not present in the original TalNet architecture.

## 📊 Datasets

The project works with multiple time series datasets:
- **Exchange Rate Data**: Currency exchange rates for multiple currencies
- **Electricity Consumption Data**
- **Solar Power Generation Data**
- **Traffic Flow Data**

### Exchange Rate Dataset Details
- **Format**: Time series data of multiple currency exchange rates
- **Features**: Multiple currency pairs
- **Frequency**: Daily data
- **Storage**: Compressed format (.gz)

## 🏗️ Model Architectures

### 1. Stacked Bidirectional LSTM
A deep learning model featuring:
- Bidirectional LSTM layers for capturing temporal dependencies
- Layer normalization for stable training
- Configurable stack depth
- Dropout for regularization
- Dense layers for final prediction

Key Parameters:
```python
- hidden_dim: Hidden layer dimensions
- dropout_rate: Dropout probability
- n_stack: Number of stacked LSTM layers
- bi: Boolean for bidirectional setup
```

### 2. TalNetV2 (Original Contribution)
A novel architecture that significantly enhances the original TalNet model. This custom-built implementation introduces several innovative features:
- **Advanced Node Embeddings**: Custom node embedding layer for enhanced feature representation
- **Causal Temporal Convolution**: Specialized convolution layer for preserving temporal causality
- **Hybrid Architecture**: Unique combination of LSTM and Transformer components
- **Multi-head Self-attention**: Advanced attention mechanism for capturing complex dependencies
- **Hierarchical Processing**: Layered approach combining temporal and spatial features
- **Adaptive Layer Normalization**: Improved training stability and performance
- **Flexible Stack Configuration**: Configurable transformer blocks for model scaling

The architecture represents a significant improvement over the original TalNet, specifically designed for complex multivariate time series forecasting tasks.

Key Components:
```python
- n_heads: Number of attention heads
- node_emb_dim: Node embedding dimensions
- temporal_conv: Causal convolution layer
- transformer_blocks: Self-attention and feed-forward layers
```

## 🛠️ Implementation Details

### Data Processing
1. **Window Creation**
   - Sliding window approach
   - Configurable window size and forecast horizon
   - TensorFlow data pipeline optimization

2. **Data Split**
   - Train-test split ratio: 80:20
   - Batch processing with prefetch

### Model Configuration
```python
Common Parameters:
- window_size = 168
- forecast_horizon = 24
- batch_size = 8
```

### Training Setup
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Early Stopping**: 
  - Monitor: validation MAE
  - Patience: 5 epochs
  - Best weights restoration

## 📈 Evaluation Metrics

The models are evaluated using comprehensive metrics:
1. **R²**: Coefficient of determination
2. **RMSE**: Root Mean Square Error
3. **MAE**: Mean Absolute Error
4. **MSE**: Mean Square Error
5. **SMAPE**: Symmetric Mean Absolute Percentage Error
6. **RSE**: Root Square Error
7. **CORR**: Correlation coefficient

Evaluation is performed at different prediction horizons:
- 3-step ahead
- 6-step ahead
- 12-step ahead
- 24-step ahead

## 💾 Model Persistence

Models can be saved and loaded using TensorFlow's Keras API:
```python
# Save model
model.save('path/to/model.keras')

# Load model
model = load_model('path/to/model.keras', custom_objects={'TalNetV2': TalNetV2})
```

## 📊 Visualization Features

The project includes comprehensive visualization tools:
1. Time series plots
2. Correlation heatmaps
3. Rolling statistics
4. Seasonal decomposition
5. Training history plots (loss and metrics)
6. Day-of-week analysis
7. Distribution histograms

## 🔍 Exploratory Data Analysis

The notebooks include detailed EDA:
- Temporal pattern analysis
- Correlation analysis between variables
- Rolling statistics computation
- Seasonal decomposition
- Distribution analysis
- Day-of-week patterns

## 🚀 Getting Started

1. **Environment Setup**
   ```python
   # Required Libraries
   import numpy as np
   import pandas as pd
   import tensorflow as tf
   import matplotlib.pyplot as plt
   import seaborn as sns
   ```

2. **Data Preparation**
   ```python
   # Load and preprocess data
   df = pd.read_csv('path/to/data.gz', compression='gzip')
   ```

3. **Model Training**
   ```python
   # Create and train model
   model = TalNetV2(params...)
   model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   model.fit(train_dataset, validation_data=test_dataset, epochs=50)
   ```

## 📈 Performance Analysis

The models are evaluated on multiple horizons with metrics including:
- R² score for goodness of fit
- RMSE for prediction accuracy
- MAE for absolute error measurement
- SMAPE for percentage error
- Correlation coefficients

## 🤝 Contributing

Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit pull requests
4. Report issues
5. Suggest improvements

## 📝 License

This project is open-source and available for academic and research purposes.

## 🔗 References

- TalNet architecture
- Time series forecasting methodologies
- Deep learning for temporal data
- Multi-horizon prediction techniques