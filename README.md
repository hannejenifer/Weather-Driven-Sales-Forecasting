# WEATHER-DRIVEN SALES FORECASTING & PRICE SENSITIVITY ANALYSIS (2013–2024)
This project focuses on weather-driven sales forecasting for both perishable and non-perishable goods, integrating historical sales data with weather variables, import trends, and price sensitivity factors. The objective is to improve demand prediction accuracy by capturing external drivers that significantly influence consumer purchasing behavior.

The system preprocesses multi-source data, performs feature engineering on weather and economic indicators, and applies machine learning–based forecasting models to identify demand patterns under varying climatic and market conditions. Special emphasis is placed on understanding how temperature, rainfall, import volume, and price fluctuations impact short-term and long-term sales.

The insights generated from this project support inventory optimization, reduced wastage for perishables, improved supply chain planning, and data-driven pricing strategies. The solution is designed to be modular, extensible, and applicable to real-world retail and distribution scenarios.

# DASHBOARD IMAGES

![image alt](https://github.com/hannejenifer/Weather-Driven-Sales-Forecasting/blob/97f4a036c572873c2fc94c5b39aa091488439ff0/Model%20Evaluation%20Metrics.png)

![image alt](https://github.com/hannejenifer/Weather-Driven-Sales-Forecasting/blob/97f4a036c572873c2fc94c5b39aa091488439ff0/Density%20Plot%20of%20Actual%20vs%20Predicted%20Prices.png)

![image alt](https://github.com/hannejenifer/Weather-Driven-Sales-Forecasting/blob/97f4a036c572873c2fc94c5b39aa091488439ff0/Distribution%20of%20Prediction%20Errors.png)

![image alt](https://github.com/hannejenifer/Weather-Driven-Sales-Forecasting/blob/main/Trend%20of%20Total%20Imported%20Value%20Over%20Years.png)

![image alt](https://github.com/hannejenifer/Weather-Driven-Sales-Forecasting/blob/main/Trend%20of%20Total%20Imported%20Value.png)

# Abstract
In the competitive restaurant industry, success hinges on accurate demand forecasting and optimal pricing strategies. Weather factors such as temperature, rain, and seasonal changes significantly impact consumer behavior, particularly affecting sales of goods categorized as highly perishable (e.g., fresh produce, dairy), perishable (e.g., bread, meat), and non-perishable (e.g., canned goods, dry ingredients). The project leverages historical weather patterns, ingredient prices, and machine learning to forecast sales and implement dynamic pricing, with a focus on import status and price sensitivity. Using Random Forest and ARIMA models, the system conducts time series analysis for precise demand prediction. Built with Python and utilizing libraries like scikit-learn and pandas, the project processes real-time weather and ingredient price data to enable timely price adjustments and inventory management. The data-driven approach empowers restaurants to maximize revenue, optimize menu pricing, and enhance customer satisfaction in a fluctuating market.

# EXECUTIVE SUMMARY
This project develops a machine-learning-based sales and price forecasting system by integrating historical weather data, FAOSTAT commodity prices, import dependency, and price sensitivity classifications. Using Random Forest regression, the model captures the influence of climatic variables on food prices while providing actionable insights for inventory planning, import strategy, and pricing decisions, particularly for perishable and non-perishable goods.

# BUSINESS PROBLEM
Retailers and supply chain planners struggle to:
- Accurately forecast prices under changing weather conditions
- Manage import dependency risks for food commodities
- Reduce wastage in highly perishable goods
- Adapt pricing strategies based on consumer price sensitivity
- Traditional forecasting methods ignore external demand drivers such as weather volatility and import patterns, leading to suboptimal decisions.

# NORTH STAR METRICS
- Price Forecast Accuracy (R² Score)
- Mean Absolute Error (MAE)
- Prediction Error Distribution
- Stability of Forecasts Across Perishable Categories

# SUPPORTING DIMENSIONS
- Average Temperature (tavg)
- Minimum & Maximum Temperature (tmin, tmax)
- Precipitation (prcp)
- Import Status (Imported vs Non-Imported)
- Price Sensitivity (Low / Medium / High)
- Perishability Category (Perishable / Non-Perishable)

# METHODOLOGY
**1. Data Collection**
- Weather data (1990–2022, Chennai region)
- FAOSTAT commodity price data
- Import and price sensitivity datasets

**2. Data Preprocessing**
- Temporal alignment using Year extraction
- Feature selection and missing value handling
- Dataset merging across multiple sources

**3. Modeling**
- Random Forest Regressor for price prediction
- Train–validation split (80–20)
- Ingredient-level modeling for granular forecasts

**4. Evaluation**
- MAE, MSE, R² metrics
- Residual and error distribution analysis
- Feature importance interpretation

**5. Visualization**
- Heatmaps for price sensitivity
- Trend analysis for imports and perishables
- Distribution and density plots for predictions

# SKILLS
- Python
- Machine Learning
- Data Visualization
- Feature Engineering
- Exploratory Data Analysis
- Business Analytics & Forecasting
- Supply Chain & Pricing Insights

# RESULTS
- Weather variables significantly influence commodity pricing
- Temperature metrics (tavg, tmax) dominate feature importance
- Imported goods exhibit higher price volatility
- Highly perishable items show sharper sensitivity to climatic changes
- The model successfully forecasts 2024 prices using projected weather inputs

# BUSINESS RECOMMENDATIONS
- Increase buffer inventory for high-sensitivity, imported perishables
- Use weather-based dynamic pricing during extreme climatic conditions
- Reduce dependency on imports for items with consistently high volatility
- Segment products by sensitivity to improve promotional effectiveness

# NEXT STEPS
- Incorporate real-time weather APIs
- Extend models to LSTM for temporal forecasting
- Add demand quantity prediction alongside price
- Build an interactive Streamlit dashboard
- Expand analysis to multiple geographic regions

# SUMMARY AND INSIGHTS
This project demonstrates how external environmental and economic factors can materially improve sales and price forecasting. By combining weather intelligence with import and sensitivity analysis, the model moves beyond traditional forecasting and supports data-driven operational and strategic decisions.

# KEY INSIGHTS
- Weather is a non-negotiable driver of food price behavior
- Imported items are structurally more volatile than domestic goods
- Perishable goods require short-horizon, weather-aware forecasting
- Feature importance improves model transparency and trust
- Price sensitivity segmentation enhances business decision-making
