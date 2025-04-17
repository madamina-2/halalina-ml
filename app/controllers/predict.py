from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from ..services.risk_profile_service import calculate_risk_profile

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
@jwt_required()  # Only accessible if there is a valid JWT
def predict():
    """Route to get prediction and risk profile."""
    data = request.get_json()

    try:
        response = calculate_risk_profile(data)
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
