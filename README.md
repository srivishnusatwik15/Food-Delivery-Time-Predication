# Food-Delivery-Time-Prediction

This project aims to predict the delivery time for food orders based on various features. The dataset used for this analysis is stored in the file "food_delivery.csv".

## Getting Started

To get started with this project, you can follow these steps:

1. Clone the repository: `git clone <repository_url>`
2. Install the required dependencies: `pip install numpy pandas matplotlib seaborn plotly scikit-learn xgboost`
3. Run the code in a Python environment: `python food_delivery_analysis.py`

## Description

The analysis is performed using Python and several data analysis libraries, including NumPy, Pandas, Matplotlib, Seaborn, and Plotly. The dataset is loaded into a Pandas DataFrame and explored to gain insights.

### Data Exploration

- Basic information about the dataset is displayed using `df.head()`, `df.tail()`, `df.shape`, `df.columns`, `df.duplicated().sum()`, `df.isnull().sum()`, `df.info()`, `df.describe()`, and `df.nunique()`.
- Categorical columns are identified using `object_columns = df.select_dtypes(include='object').columns`.
- Numerical columns are identified using `numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns`.
- The unique values and value counts for the "Type_of_order" and "Type_of_vehicle" columns are displayed.
- Count plots, pie charts, bar plots, box plots, violin plots, histograms, and density plots are created to visualize the distributions and relationships between variables.

### Feature Engineering

- Time components (hour of day, day of week, month of year) are extracted from the "Time_taken(min)" column.
- A function is defined to calculate the distance between latitude and longitude coordinates using the Haversine formula.
- The distance feature is created based on the coordinates of the restaurant and delivery location.
- The "Delivery_person_Age" column is categorized into age groups using binning.
- The average ratings for each delivery person are calculated and assigned to the "avg_ratings" column.
- Binary encoding is applied to the "Type_of_order" and "Type_of_vehicle" columns.
- An interaction feature is created by multiplying "Time_taken(min)" and "Delivery_person_Ratings".

### Data Preprocessing

- Irrelevant columns are dropped from the dataset using `df.drop(columns=columns_to_drop)`.
- Feature scaling is performed on selected numerical features using the StandardScaler.
- One-hot encoding is applied to the "age_category" feature using the ColumnTransformer.

### Model Training and Evaluation

- The dataset is split into training and testing sets using train_test_split.
- Linear Regression, Decision Tree Regressor, and XGBoost Regressor models are trained and evaluated.
- Predictions are made on the test set, and evaluation metrics such as RMSE, R-squared, and MSE are calculated for each model.

## Results

The results of the analysis are as follows:

- Linear Regression:
  - Root Mean Squared Error: \<RMSE value>
  - R-squared Score: \<R-squared value>
  - Mean Squared Error: \<MSE value>

- Decision Tree Regressor:
  - R-squared Score: \<R-squared value>
  - Mean Squared Error: \<MSE value>

## Conclusion

In this project, we explored a food delivery dataset and built predictive models to estimate the delivery time. The analysis revealed the relationships between different features and the target variable. The models were evaluated using various metrics, and the results indicate the effectiveness of the models in predicting delivery time.
