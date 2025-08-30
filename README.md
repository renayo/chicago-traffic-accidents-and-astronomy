# Chicago Traffic Accidents & Astronomical Patterns
## A Data-Driven Analysis of Celestial Correlations with Road Safety

---

## Executive Overview

This analysis explores potential correlations between astronomical positions and hourly traffic accident volumes in Chicago. Using machine learning techniques applied to nearly 10 years of accident data (84,987 hourly records), we developed predictive models incorporating 17 astronomical features. Our best-performing model achieved an R² score of 0.642, demonstrating notable predictive capability while raising intriguing questions about temporal patterns in accident data.

---

## Key Findings

### Performance Metrics
- **Model Accuracy**: R² = 0.642 (explaining 64.2% of variance in accident patterns)
- **Prediction Error**: RMSE = 4.97 accidents per hour
- **Dataset Scale**: 84,987 hourly observations spanning approximately 9.7 years
- **Average Accident Rate**: 11.5 accidents per hour (Chicago metropolitan area)
- **Peak Hour Maximum**: 141 accidents recorded in a single hour

### Visual Analysis

**Distribution of Actual Hourly Accidents**
<img width="480" height="291" alt="Actual accident distribution" src="https://github.com/user-attachments/assets/b236f895-e4f8-4bc4-a211-fe95afa4bc59" />

**Model Predictions Distribution**
<img width="480" height="291" alt="Predicted accident distribution" src="https://github.com/user-attachments/assets/07c93725-3a14-4141-b75e-20e2105e3887" />

**Prediction Accuracy (Residuals)**
<img width="480" height="291" alt="Prediction accuracy" src="https://github.com/user-attachments/assets/b5a81338-86e8-4e8f-8611-35e1c9ba4c9d" />

---

## Methodology

### Astronomical Features Analyzed

We computed 17 astronomical measurements for each hour of accident data:

#### Primary Solar-Lunar Features
- **Ascendant**: The ecliptic degree rising on the eastern horizon at Chicago's coordinates
- **Sun Longitude**: Solar position along the ecliptic (0-360°)
- **Moon Longitude**: Lunar ecliptic position
- **Moon Distance**: Earth-Moon distance in Earth radii
- **Moon Phase**: Lunar illumination percentage (0-1 scale)

#### Planetary Positions (Ecliptic Longitude)
- **Inner Planets**: Mercury, Venus, Mars
- **Gas Giants**: Jupiter, Saturn
- **Outer Planets**: Uranus, Neptune, Pluto

#### Trans-Neptunian and Minor Bodies
- **Eris**: Dwarf planet in the scattered disc
- **Sedna**: Distant trans-Neptunian object
- **Pholus & Nessus**: Centaur asteroids with eccentric orbits

---

## Model Performance Analysis

### Random Forest Regressor (Optimal Model)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Test R² | 0.642 | Strong predictive capability |
| Training R² | 0.746 | Minimal overfitting observed |
| Test RMSE | 4.97 | ±5 accidents typical error range |
| Test MAE | 3.62 | Average absolute deviation |
| Cross-Validation RMSE | 5.01 | Consistent across data splits |

### Gradient Boosting Regressor (Alternative Model)

| Metric | Value |
|--------|-------|
| Test R² | 0.603 |
| Test RMSE | 5.23 |
| Cross-Validation RMSE | 5.26 |

---

## Feature Importance Analysis

### Most Influential Astronomical Variables

| Rank | Feature | Importance | Key Insight |
|------|---------|------------|-------------|
| 1 | **Ascendant** | 32.8% | Strongest predictor; changes every ~2 hours |
| 2 | **Sun Longitude** | 25.2% | Captures daily and seasonal cycles |
| 3 | **Uranus Longitude** | 13.8% | Unexpected significance; suggests long-term patterns |
| 4 | **Mercury Longitude** | 7.6% | Moderate influence on predictions |
| 5 | **Venus Longitude** | 2.6% | Minor but measurable effect |
| 6 | **Nessus Longitude** | 2.5% | Centaur asteroid shows statistical relevance |
| 7 | **Pholus Longitude** | 2.1% | Secondary centaur influence |
| 8 | **Jupiter Longitude** | 1.8% | Minimal but consistent correlation |
| 9 | **Moon Distance** | 1.7% | Lower than anticipated influence |
| 10 | **Eris Longitude** | 1.7% | Dwarf planet shows slight correlation |

### Statistical Observations

1. **Dominant Predictors**: Ascendant and Sun longitude collectively account for 58% of model importance
2. **Unexpected Pattern**: Uranus's high importance (13.8%) warrants further investigation
3. **Lunar Factors**: Moon phase and distance showed surprisingly low predictive value
4. **Minor Bodies**: Measurable influence from trans-Neptunian objects despite extreme distances

---

## Interpretation & Implications

### Scientific Context

While our model demonstrates significant predictive accuracy, it's crucial to note that **correlation does not imply causation**. The astronomical features likely serve as sophisticated temporal markers rather than causal factors. Of course, most astrologers would not say otherwise. The high importance of the Ascendant and Sun longitude particularly suggests these features effectively encode:

- Time-of-day variations in traffic patterns
- Seasonal fluctuations in accident rates
- Complex periodic cycles in urban mobility

### Practical Applications

1. **Resource Allocation**: Emergency services could use predictions to optimize deployment during high-risk periods
2. **Pattern Recognition**: Identified cycles may reveal underlying traffic flow dynamics
3. **Predictive Framework**: Model provides baseline for comparing other predictive approaches

### Areas for Further Research

1. **Uranus Correlation Investigation**: The 13.8% importance of Uranus longitude merits dedicated analysis
2. **Mechanism Exploration**: Decompose which temporal patterns align with specific astronomical cycles
3. **Geographic Validation**: Test model transferability to other metropolitan areas
4. **Feature Engineering**: Explore planetary aspects and angular relationships

---

## Technical Implementation

### Model Configuration
- **Algorithm**: Random Forest with 200 estimators
- **Validation**: 5-fold cross-validation
- **Data Split**: 80% training (67,990 records), 20% testing (16,997 records)
- **Feature Scaling**: StandardScaler normalization

### Deliverables
- Trained model (`accident_astro_model.pkl`)
- Feature dataset (`astrological_features.csv`)
- Prediction results (`predictions.csv`)
- Complete codebase and documentation

---

## Data Sources & Resources

- **Source Data**: [Chicago Traffic Crashes Database](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data)
- **Complete Project Files**: [Dropbox Repository](https://www.dropbox.com/scl/fo/wbfaw4ldii5bq8iimeb6a/ALloKkGhNRJmlkQIG6UIKLA?rlkey=ep82d3o937wq2fuy4wfweeept&st=6i0enwk6&dl=0)

---

## Comparison to Time Series Best Fit Model

The best fit time series model, specified as MAProcess[11.5993, {}, 68.9079], is an MA (Moving Average) model used for forecasting time series data. The parameters are as follows:

The first parameter, 11.5993, represents the mean of the time series.

The empty braces {} indicate that this is a moving average process of order zero (q=0).

The third parameter, 68.9079, is the variance of the white noise component.

An MA(0) model is a simplified case where the time series is just a white noise process with a constant mean. This means the model doesn't use past values or errors to predict future values; it essentially just predicts the mean with a certain variance. This makes it a very weak model for forecasting anything beyond a constant. Therefore, it's not a strong candidate for making accurate predictions.

### Random Forest Model
The random forest model provides several performance metrics that allow for a more robust evaluation:

*R-squared Score (0.7929):* This score indicates that the model explains approximately 79.3% of the variance in the target variable. This is a solid score, suggesting the model has a high degree of predictive power. A score closer to 1 is better.

*RMSE (32.59):* The Root Mean Square Error is the standard deviation of the residuals (prediction errors). It measures the average magnitude of the errors. A lower RMSE is better.

*MAE (23.92):* The Mean Absolute Error is another measure of error magnitude. It's less sensitive to outliers than RMSE. A lower MAE is also better.

*Cross-Val RMSE (34.72 ±8.63):* This metric is crucial because it indicates the model's generalization ability on unseen data. The value is a measure of the average RMSE across multiple folds of cross-validation, and the standard deviation (±8.63) gives a sense of the model's stability. The fact that the cross-validation RMSE is close to the regular RMSE suggests the model is not overfitting and will perform similarly on new data.

### Evaluation of Comparison
The random forest model provides a quantitative measure of its predictive ability, explaining a significant portion of the variance in the data and showing good performance on unseen data. In contrast, the provided time series model is a simple white noise process that lacks the ability to capture any underlying patterns, making it a poor choice for prediction. Thus, the random forest is demonstrably superior for this task.

## Conclusions

This analysis reveals statistically significant correlations between astronomical positions and Chicago traffic accident patterns, with our model achieving 64.2% variance explanation. While the mechanisms underlying these correlations remain to be fully understood, the model's predictive accuracy offers practical value for traffic safety management and resource planning.

The unexpected importance of certain astronomical features, particularly Uranus's position, presents intriguing questions for future research. Whether these patterns reflect complex temporal cycles, subtle environmental influences, or statistical artifacts, they demonstrate the value of exploring unconventional approaches to traffic safety analysis.

---

*Analysis Period: ~9.7 years | Sample Size: 84,987 hourly records | Model Performance: R² = 0.642*  
*Chicago, Illinois Metropolitan Area Traffic Data*
