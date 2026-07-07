# 🚢 Maritime Vessel Trajectory Prediction using Machine Learning, Geospatial Analytics & Time-Series Analysis

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![Folium](https://img.shields.io/badge/Folium-Geospatial-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

# Executive Summary

Modern maritime logistics generates millions of Automatic Identification System (AIS) records every day. Extracting meaningful operational insights from these high-volume geospatial datasets is challenging due to vessel movement complexity, trajectory variability, and noisy sensor observations.

This project presents an end-to-end machine learning pipeline for maritime vessel trajectory prediction using more than **1.09 million AIS records**. The solution combines data preprocessing, feature engineering, exploratory data analysis, geospatial analytics, anomaly detection, time-series analysis, and machine learning to predict future vessel positions while generating actionable operational insights.

The project demonstrates how historical vessel movement patterns can be transformed into predictive intelligence for maritime monitoring and operational decision-making.

---

# Business Problem

Maritime organizations continuously receive massive streams of AIS data from thousands of vessels operating across global waters.

Without advanced analytics, it becomes difficult to:

- Predict future vessel positions
- Analyze traffic patterns
- Detect unusual vessel behavior
- Monitor operational efficiency
- Support maritime decision-making

Traditional monitoring approaches are reactive and depend heavily on manual observation. A predictive analytics solution enables organizations to anticipate vessel movement and improve operational awareness.

---

# Project Objectives

The primary objectives of this project are:

- Build a complete AIS data processing pipeline.
- Clean and validate over one million vessel records.
- Engineer predictive geospatial and temporal features.
- Analyze vessel movement using time-series techniques.
- Detect abnormal vessel behavior.
- Train a machine learning model for trajectory prediction.
- Evaluate forecasting performance using a realistic prediction horizon.
- Produce recruiter-friendly visualizations and project documentation.

---

# Key Features

- End-to-End Data Analytics Pipeline
- Data Quality Assessment & Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Geospatial Analytics
- Maritime Traffic Visualization
- Time-Series Analysis
- Maritime Anomaly Detection
- Random Forest Trajectory Prediction
- Forecast Horizon Evaluation (1-Step vs. 5-Step)
- Feature Importance Analysis
- Model Export using Joblib

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.14 |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Geospatial Analytics | Folium, Branca, Geopy |
| Scientific Computing | SciPy |
| Model Serialization | Joblib |
| Development Environment | VS Code, Jupyter Notebook |

---

# 📊 Dataset Overview

| Metric | Value |
|--------|------:|
| Raw AIS Records | 1,098,966 |
| Cleaned Records | 1,098,888 |
| Duplicate Records Removed | 78 |
| Unique Vessels | 12,783 |
| Prediction Horizon | 5 AIS Messages |
| Machine Learning Algorithm | Random Forest Regressor |

### Dataset Features

The dataset contains maritime Automatic Identification System (AIS) information including:

- Vessel Identifier (MMSI)
- Timestamp
- Latitude & Longitude
- Speed Over Ground (SOG)
- Course Over Ground (COG)
- Heading
- Vessel Dimensions
- Draft
- Vessel Type
- Cargo Information
- Destination Coordinates

---

# 🏗️ Project Architecture

```text
                     AIS Dataset
                          │
                          ▼
               Data Quality Assessment
                          │
                          ▼
                  Data Cleaning Pipeline
                          │
                          ▼
                 Feature Engineering
                          │
                          ▼
             Exploratory Data Analysis
                          │
                          ▼
               Geospatial Analytics
                          │
                          ▼
                Time-Series Analysis
                          │
                          ▼
             Maritime Anomaly Detection
                          │
                          ▼
         Trajectory Prediction (Random Forest)
                          │
                          ▼
            Forecast Horizon Evaluation
                          │
                          ▼
             Business Insights & Reports
```

---

# 📁 Project Structure

```text
Maritime-Vessel-Trajectory-Prediction-ML/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebook/
│   └── Maritime_Vessel_Trajectory_Prediction.ipynb
│
├── outputs/
│   ├── figures/
│   ├── maps/
│   ├── models/
│   └── reports/
│
├── src/
│
├── docs/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🔄 Project Workflow

```text
Raw AIS Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Geospatial Analysis
      │
      ▼
Time-Series Analysis
      │
      ▼
Anomaly Detection
      │
      ▼
Trajectory Prediction
      │
      ▼
Forecast Horizon Comparison
      │
      ▼
Business Insights
```

---

# ⚙️ Methodology

The project follows a structured end-to-end data analytics and machine learning workflow to transform raw AIS data into predictive maritime intelligence.

### Phase 1 — Data Acquisition
- Loaded over **1.09 million AIS records**.
- Parsed timestamps and validated schema consistency.
- Performed initial exploratory inspection.

### Phase 2 — Data Cleaning
- Removed duplicate records.
- Removed duplicate MMSI + Timestamp combinations.
- Validated latitude and longitude ranges.
- Standardized missing values.
- Verified data integrity before analysis.

### Phase 3 — Feature Engineering
Engineered multiple predictive features to improve model performance, including:

- Hour of Day
- Month
- Time Difference Between Consecutive AIS Messages
- Previous Speed (Prev_SOG)
- Previous Course (Prev_COG)
- Previous Heading (Prev_Heading)
- Speed Categories
- Movement Status

### Phase 4 — Exploratory Data Analysis
Performed exploratory analysis to understand:

- Vessel traffic distribution
- Vessel movement patterns
- Speed distribution
- Temporal traffic trends
- Geographic vessel density

### Phase 5 — Geospatial Analytics
Used vessel coordinates to:

- Visualize vessel locations
- Analyze vessel movement
- Explore maritime traffic patterns
- Generate interactive route maps

### Phase 6 — Maritime Anomaly Detection
Identified operational anomalies such as:

- High-Speed Vessels
- Sudden Acceleration
- Sudden Heading Changes
- Long AIS Transmission Gaps
- GPS Position Jumps

### Phase 7 — Trajectory Prediction
Built a supervised machine learning model to predict future vessel positions using historical movement characteristics.

---

# 🔍 Data Quality Assessment

Data quality plays a critical role in maritime analytics because prediction accuracy depends on reliable AIS observations.

The following validation checks were performed before model development.

| Quality Check | Status |
|--------------|--------|
| Duplicate Records Removed | ✅ |
| Duplicate MMSI + Timestamp Removed | ✅ |
| Invalid Latitude Detection | ✅ |
| Invalid Longitude Detection | ✅ |
| Missing Value Validation | ✅ |
| Timestamp Verification | ✅ |

Final cleaned dataset contained **1,098,888** high-quality AIS observations.

---

# 🧠 Feature Engineering Strategy

Feature engineering was designed to capture both spatial and temporal vessel movement.

### Spatial Features

- Latitude
- Longitude
- Draft
- Length
- Width

### Motion Features

- Speed Over Ground (SOG)
- Course Over Ground (COG)
- Heading

### Historical Features

- Previous Speed
- Previous Course
- Previous Heading

### Temporal Features

- Hour
- Month
- Time Difference Between Messages

This combination enables the model to learn vessel movement patterns rather than relying solely on current location.

---

# 🤖 Machine Learning Pipeline

The prediction pipeline consists of the following stages.

```text
Cleaned AIS Data
        │
        ▼
Feature Engineering
        │
        ▼
Forecast Target Creation
(5-Step Ahead)
        │
        ▼
GroupShuffleSplit
(by MMSI)
        │
        ▼
Random Forest Regressor
        │
        ▼
Prediction
        │
        ▼
Performance Evaluation
```

---

# 🎯 Model Selection

A **Random Forest Regressor** was selected because:

- Handles non-linear relationships effectively.
- Requires minimal feature scaling.
- Performs well on structured tabular data.
- Provides feature importance scores.
- Offers strong baseline performance with limited hyperparameter tuning.

While advanced sequence models such as LSTM or Transformer architectures may capture longer temporal dependencies, Random Forest provides an excellent balance of predictive performance, interpretability, and computational efficiency for this project.

---

# 📈 Forecast Horizon Strategy

Rather than predicting only the immediate next AIS transmission, the final model forecasts vessel positions **five AIS messages into the future**.

This design choice creates a more challenging and operationally meaningful prediction task by reducing dependence on immediate spatial continuity and encouraging the model to learn broader movement trends.

---

# 📊 Model Performance

## Trajectory Prediction Results

| Target | MAE | RMSE | R² Score |
|---------|------:|------:|------:|
| Future Latitude | 0.031422 | 0.273736 | 0.998572 |
| Future Longitude | 0.029123 | 0.359736 | 0.999550 |

The high coefficient of determination indicates that the model effectively captures vessel movement patterns while maintaining low prediction error across both latitude and longitude.

---

# ⭐ Feature Importance

The Random Forest model identified the following variables as the strongest predictors of future vessel position.

| Rank | Feature | Importance |
|------:|---------|-----------:|
| 1 | Longitude | 0.908950 |
| 2 | Latitude | 0.091001 |
| 3 | Previous Speed | 0.000010 |
| 4 | Speed Over Ground | 0.000010 |
| 5 | Draft | 0.000005 |
| 6 | Length | 0.000005 |
| 7 | Hour | 0.000004 |
| 8 | Previous Heading | 0.000004 |
| 9 | Width | 0.000003 |
| 10 | Heading | 0.000003 |

The dominance of longitude and latitude is expected because consecutive AIS transmissions exhibit strong spatial continuity. Auxiliary movement and vessel characteristics provide additional contextual information for trajectory prediction.

---

# 💡 Business Insights

Key insights generated from the analysis include:

- AIS data can be transformed into predictive operational intelligence through structured preprocessing and feature engineering.
- Vessel movement exhibits strong spatial continuity, making historical coordinates highly informative for short-term trajectory prediction.
- Geospatial analysis helps identify traffic concentration and movement patterns.
- Anomaly detection can highlight potentially unusual vessel behavior for further operational review.
- Forecast horizon selection has a direct impact on prediction difficulty and model performance, demonstrating the importance of evaluation strategy in time-series forecasting.

---

# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/Maritime-Vessel-Trajectory-Prediction-ML.git
```

Navigate to the project directory:

```bash
cd Maritime-Vessel-Trajectory-Prediction-ML
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
notebook/
Maritime_Vessel_Trajectory_Prediction.ipynb
```

Run all notebook cells sequentially.

The notebook automatically performs:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Geospatial Analysis
- Time-Series Analysis
- Maritime Anomaly Detection
- Machine Learning
- Forecast Horizon Evaluation

Generated outputs are saved inside:

```
outputs/
```

---

# 📂 Generated Outputs

The project automatically creates:

```
outputs/

├── figures/
│
├── reports/
│
├── models/
│
└── maps/
```

Including:

- Model Evaluation Reports
- Feature Importance
- Forecast Horizon Comparison
- Saved Random Forest Model
- Geospatial Outputs

---

# 📸 Project Screenshots

The following project outputs are recommended for inclusion inside the **docs/images/** directory.

| Screenshot | Description |
|------------|-------------|
| Architecture Diagram | Overall project workflow |
| EDA Dashboard | Exploratory Data Analysis |
| Feature Importance | Random Forest Feature Importance |
| Forecast Horizon Comparison | 1-Step vs 5-Step Forecast |
| Geospatial Map | Vessel Movement Visualization |

> After uploading the screenshots, update the README with embedded images using Markdown.

Example:

```markdown
![Feature Importance](docs/images/feature_importance.png)
```

---

# 🚀 Future Improvements

Potential enhancements include:

- Implement XGBoost and LightGBM for performance comparison.
- Evaluate sequence-based models such as LSTM or Transformer architectures.
- Deploy the prediction model as a web application using Streamlit or FastAPI.
- Integrate live AIS data streams for real-time trajectory prediction.
- Build an interactive maritime dashboard for operational monitoring.
- Optimize feature engineering with rolling statistics and lag-based features.
- Extend the forecasting horizon for long-range vessel movement prediction.

---

# 📚 Learning Outcomes

This project strengthened practical experience in:

- Large-scale data preprocessing
- Data quality assessment
- Feature engineering
- Geospatial analytics
- Time-series analysis
- Machine learning for regression
- Model evaluation
- Technical documentation
- GitHub project organization

---

# 🤝 Contributing

Contributions, suggestions, and constructive feedback are welcome.

If you find this project useful, consider giving it a ⭐ on GitHub.

---

# 📄 License

This project is licensed under the MIT License.

See the **LICENSE** file for additional details.

---

# 👨‍💻 Author

**Arpit Mukherjee**

AI-Assisted Data Analyst | Python | SQL | Power BI | Machine Learning | Geospatial Analytics

If you'd like to connect or discuss this project:

- LinkedIn: *(Add your LinkedIn URL)*
- GitHub: *(Add your GitHub Profile URL)*

---

# ⭐ Acknowledgement

This project was developed as a portfolio project to demonstrate practical applications of machine learning, geospatial analytics, and time-series analysis in the maritime domain.

The implementation focuses on building an end-to-end predictive analytics pipeline using publicly available AIS vessel movement data for educational and portfolio purposes.
