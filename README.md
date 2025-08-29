# Predict-House-Price

🏡 House Price Prediction using Linear Regression
📌 Project Overview

This project develops a Linear Regression model to predict house prices using features such as size, number of rooms, location, and other property attributes.
The dataset is taken from Kaggle: House Prices – Advanced Regression Techniques
.

The workflow includes:

Data preprocessing (handling missing values, categorical encoding, scaling)

Training a Linear Regression model

Evaluating model performance with metrics like MAE, RMSE, and R²

Visualizing actual vs predicted house prices

📂 Dataset

The dataset (train.csv) contains information about residential homes in Ames, Iowa.

Target Variable: SalePrice (house price)

Features include:

LotArea, OverallQual, YearBuilt, TotalBsmtSF, GrLivArea

BedroomAbvGr, GarageCars, FullBath

Neighborhood, HouseStyle, and other categorical variables

⚙️ Requirements

Install the following dependencies before running the project:

pip install pandas numpy matplotlib seaborn scikit-learn

🚀 Steps to Run the Project

Clone the repository / Download the code

git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction


Download the dataset

Go to the Kaggle dataset page

Download train.csv

Place it in the project folder

Run the Python script

python house_price_prediction.py


Output

Model performance metrics: MAE, MSE, RMSE, R²

Scatter plot of Actual vs Predicted house prices

📊 Example Results

MAE (Mean Absolute Error): ~ 22,000

RMSE (Root Mean Squared Error): ~ 34,000

R² Score: ~ 0.82 (indicating good fit)

(Results may vary depending on preprocessing and train-test split)

📈 Visualization

The scatter plot below shows the Actual vs Predicted House Prices:

   (points close to the diagonal line indicate better prediction accuracy)

🔮 Future Improvements

Apply Regularization Models (Ridge, Lasso, ElasticNet)

Use Feature Engineering (log-transform skewed data, handle outliers)

Try Tree-based models (RandomForest, XGBoost) for better accuracy

Deploy the model using Flask / Django / Streamlit
