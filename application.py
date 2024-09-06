from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
cors = CORS(app)

# Training the model
def train_model():
    # Load and prepare the data
    car = pd.read_csv(r'C:\MiniProject\Cleaned_Car_data.csv')
    
    # Print columns for debugging
    print("Columns in the dataset:", car.columns)

    if 'Price' not in car.columns:
        raise KeyError("The column 'Price' is missing from the dataset")

    X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['year', 'kms_driven']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])
        ],
        remainder='passthrough'
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Save the model
    with open(r'C:\MiniProject\LinearRegressionModel.pkl', 'wb') as file:
        pickle.dump(pipeline, file)

# Train the model
train_model()

# Load the model
with open(r'C:\MiniProject\LinearRegressionModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the car data
car = pd.read_csv(r'C:\MiniProject\Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    input_data = pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [driven],
        'fuel_type': [fuel_type]
    })

    prediction = model.predict(input_data)
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run()
