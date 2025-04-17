from flask import Flask
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from .config import Config

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize JWTManager
    JWTManager(app)

    # Enable CORS for all origins (adjust as needed)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register your routes here
    from .controllers.predict import predict_bp
    app.register_blueprint(predict_bp, url_prefix='/api')

    return app
