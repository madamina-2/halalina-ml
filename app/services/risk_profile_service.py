import os
import numpy as np
import pandas as pd
import pickle
from flask import current_app

# Open and load the model
with open('model/best_random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def calculate_risk_profile(data):
    """Calculate the risk profile and allocate portfolio percentages."""
    # Prepare the data for prediction
    new_data = pd.DataFrame([{
        'job': data['job'],
        'marital': data['marital'],
        'balance': data['balance'],
        'age_group': data['age_group'],
        'is_having_debt': data['is_having_debt']
    }])

    # Make prediction
    prediction = loaded_model.predict(new_data)
    proba = loaded_model.predict_proba(new_data)

    # Define risk profile mapping
    risk_profiles = {0: "Agresif", 1: "Moderat", 2: "Defensif"}

    # Base allocations for each profile
    base_allocations = {
        "Agresif": {"Tabungan Emas": 0.40, "SBSN": 0.30, "RDPU Syariah": 0.20, "Deposito Syariah": 0.10},
        "Moderat": {"Tabungan Emas": 0.25, "SBSN": 0.35, "RDPU Syariah": 0.25, "Deposito Syariah": 0.15},
        "Defensif": {"Tabungan Emas": 0.05, "SBSN": 0.40, "RDPU Syariah": 0.30, "Deposito Syariah": 0.25}
    }

    proba_scores = np.array(proba).flatten()
    final_allocation = {key: 0.0 for key in base_allocations["Agresif"].keys()}

    # Calculate weighted allocation
    for i, profile in enumerate(["Agresif", "Moderat", "Defensif"]):
        for product in final_allocation:
            final_allocation[product] += base_allocations[profile][product] * proba_scores[i]

    # Convert to percentage
    percent_values = {k: v * 100 for k, v in final_allocation.items()}
    floored = {k: int(v) for k, v in percent_values.items()}
    remainders = {k: percent_values[k] - floored[k] for k in percent_values}

    total_floored = sum(floored.values())
    diff = 100 - total_floored
    sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)

    # Adjust to make sure the total is 100
    for i in range(diff):
        floored[sorted_remainders[i][0]] += 1

    risk_profile = risk_profiles[int(prediction[0])]

    response = {
        "risk_profile": risk_profile,
        "floored_percentages": floored
    }
    
    return response
