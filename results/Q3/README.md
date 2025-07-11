# Gold Price Prediction Models - Question 3 Results

## Overview
This folder contains comprehensive machine learning models for gold price prediction based on the correlation analysis from Question 2.

**Best Model:** Linear_Regression  
**Test R²:** 1.0000  
**Test RMSE:** 0.0000  
**Test MAE:** 0.0000  

## Folder Structure

```
Q3/
├── README.md                           # This file
├── plot/                              # Prediction visualizations
├── report/                            # Model performance reports
├── data/                              # Results and predictions
└── models/                            # Trained model files
```

## Models Evaluated

8 machine learning models were trained and evaluated:

### Model Performance Ranking
1. **Linear_Regression** - R²: 1.0000, RMSE: 0.0000
2. **SVR_Linear** - R²: 0.9997, RMSE: 0.0802
3. **Lasso_Regression** - R²: 0.9991, RMSE: 0.1373
4. **Ridge_Regression** - R²: 0.9988, RMSE: 0.1634
5. **Gradient_Boosting** - R²: 0.9969, RMSE: 0.2621
6. **Random_Forest** - R²: 0.9953, RMSE: 0.3208
7. **Elastic_Net** - R²: 0.9865, RMSE: 0.5445
8. **SVR_RBF** - R²: 0.9112, RMSE: 1.3963


## Feature Engineering

### Market Factors (Based on Q2 Correlation Analysis)
- **GDX_Close** (Gold Miners ETF) - r=0.9755
- **PLT_Price** (Platinum Price) - r=0.7759  
- **USDI_Price** (USD Index) - r=-0.7216
- **OF_Price** (Oil Price) - r=0.7107
- **SP_close** (S&P 500) - r=-0.6843
- **USO_Close** (Oil ETF) - r=0.6357
- **OS_Price** (Silver Price) - r=0.6308

### Technical Indicators
- Moving Averages (5, 20, 60 days)
- Price Change indicators (1, 5, 20 days)
- Return indicators (1, 5, 20 days)
- Volatility (20-day rolling)
- RSI (14-day)
- Lag features (1, 2, 3, 5, 10 days)
- Rolling statistics (mean, std, min, max)

## Model Analysis

### Best Model: Linear_Regression
- **Test R²:** 1.0000 (Excellent)
- **Test RMSE:** $0.00
- **Test MAE:** $0.00
- **Test MAPE:** 0.00%
- **Overfitting:** 0.0000 (Low)

### Key Insights
1. **Model Reliability:** R² > 0.7 indicates strong predictive capability
2. **Feature Importance:** Market factors from Q2 analysis prove most valuable
3. **Generalization:** Good generalization to unseen data
4. **Prediction Accuracy:** Mean absolute percentage error of 0.0%

## Files Description

### 📊 Visualizations (plot/)
1. **01_model_performance_comparison.png** - Comprehensive model comparison
2. **02_best_model_analysis.png** - Detailed analysis of best model
3. **03_feature_importance.png** - Feature importance (for tree models)
4. **04_error_analysis.png** - Prediction error analysis

### 📋 Reports (report/)
- **model_performance_report.txt** - Detailed performance analysis

### 💾 Data (data/)
- **model_performance.csv** - All model metrics
- **prediction_results.csv** - Actual vs predicted values
- **feature_information.csv** - Feature descriptions

## Methodology

### Data Preparation
1. **Feature Selection:** Based on Q2 correlation analysis results
2. **Feature Engineering:** Technical indicators and time series features
3. **Data Splitting:** Time series split (80% train, 20% test)
4. **Scaling:** StandardScaler for all features

### Model Training
1. **Linear Models:** Linear, Ridge, Lasso, Elastic Net Regression
2. **Tree Models:** Random Forest, Gradient Boosting
3. **Support Vector Models:** Linear and RBF SVR
4. **Evaluation:** 5-fold time series cross-validation

### Performance Metrics
- **R² Score:** Coefficient of determination
- **RMSE:** Root Mean Square Error
- **MAE:** Mean Absolute Error  
- **MAPE:** Mean Absolute Percentage Error
- **Overfitting:** Train R² - Test R² difference

## Business Applications

### Investment Strategy
1. **Price Forecasting:** Reliable short-term price predictions
2. **Risk Management:** RMSE of $0.00 provides risk bounds
3. **Market Timing:** Model signals for entry/exit decisions

### Risk Assessment
- **Prediction Interval:** ±$0.00 (1 standard deviation)
- **Confidence Level:** 100.0% average accuracy
- **Maximum Error:** Monitor for prediction errors > 2 × RMSE

## Model Limitations

1. **Time Dependency:** Performance may degrade over time
2. **Market Regime Changes:** Model trained on historical relationships
3. **Black Swan Events:** Cannot predict unprecedented market shocks
4. **Feature Availability:** Requires real-time data for all input features

## Future Improvements

1. **Advanced Models:** Deep learning, LSTM networks
2. **Alternative Features:** Sentiment analysis, news data
3. **Ensemble Methods:** Combine multiple models
4. **Real-time Updates:** Online learning capabilities

---
*Generated by Gold Price Prediction System*  
*Date: 2025-07-04 23:42:04*
