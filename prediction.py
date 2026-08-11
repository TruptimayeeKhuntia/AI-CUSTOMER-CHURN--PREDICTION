import pandas as pd
from joblib import load

loaded_model = load("model_dir/housing_model.joblib")

new_house = pd.DataFrame({
    "longitude": [-122.23],
    "latitude": [37.88],
    "housing_median_age": [41],
    "total_rooms": [880],
    "total_bedrooms": [129],
    "population": [322],
    "households": [126],
    "median_income": [8.3252],
})

prediction = loaded_model.predict(new_house)

print("Prediction:", prediction[0])

prediction = loaded_model.predict(new_house)
probability = loaded_model.predict_proba(new_house)

if prediction[0] == 1:
    print("Customer is likely to CHURN ❌")
    print("Churn Probability:", probability[0][1])
else:
    print("Customer is likely to STAY ✅")
    print("Churn Probability:", probability[0][0])