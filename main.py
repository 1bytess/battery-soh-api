from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import uvicorn
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with custom docs URL
app = FastAPI(
    title="Battery Capacity Prediction API",
    description="AI-powered battery capacity prediction using BiLSTM, TCN, and LSTM models",
    version="3.0.0",
    docs_url="/v3/docs",
    redoc_url="/v3/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class BatteryData(BaseModel):
    voltage: List[float] = Field(..., description="Voltage sequence data (V)", min_length=5, max_length=200)
    current: List[float] = Field(..., description="Current sequence data (A)", min_length=5, max_length=200)    
    temperature: List[float] = Field(..., description="Temperature sequence data (°C)", min_length=5, max_length=200)
    battery_id: str = Field(..., description="Battery identifier from dataset")
    model_type: str = Field(default="lstm", description="Model to use: 'lstm', 'bilstm', or 'tcn'")
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "voltage": [4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5],
                "current": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "temperature": [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
                "battery_id": "B0005",
                "model_type": "lstm"
            }
        }
    }

class PredictionResponse(BaseModel):
    soh_percentage: float = Field(..., description="State of Health as percentage (0-100%)")
    estimated_capacity: float = Field(..., description="Estimated capacity in Ah")
    model_used: str = Field(..., description="Model that generated the prediction")
    confidence_score: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    processing_time: float = Field(..., description="Processing time in seconds")
    battery_info: Dict = Field(..., description="Information about the selected battery")
    timestamp: str
    
    model_config = {
        "protected_namespaces": ()
    }

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    available_models: List[str]
    available_batteries: Dict[str, List[str]]
    gpu_available: bool
    timestamp: str

class BatteryListResponse(BaseModel):
    train_batteries: List[str]
    test_batteries: List[str]
    total_batteries: int

# Global variables for models and scalers
models = {}
scalers = {}

# Battery database from your training
BATTERY_DATABASE = {
    'train_batteries': ['B0005', 'B0006', 'B0007', 'B0018', 'B0025', 'B0026', 'B0027', 'B0028', 
                       'B0031', 'B0032', 'B0033', 'B0034', 'B0036', 'B0039', 'B0041', 'B0042', 
                       'B0043', 'B0044', 'B0045', 'B0047', 'B0048', 'B0053', 'B0055', 'B0056'],
    'test_batteries': ['B0029', 'B0030', 'B0038', 'B0040', 'B0046', 'B0054']
}

# Battery specifications
BATTERY_SPECS = {
    'nominal_capacity': 2.0
}

def load_models_and_scalers():
    """Load trained models and scalers"""
    global models, scalers
    
    try:
        # Debug: List current working directory and available files
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        # Check for models directory
        models_dir = 'models'
        if os.path.exists(models_dir):
            logger.info(f"Models directory found: {models_dir}")
            logger.info(f"Files in models directory: {os.listdir(models_dir)}")
        else:
            logger.warning(f"Models directory not found: {models_dir}")
            models = {}
            scalers = create_default_scalers()
            return
        
        # Load models - try .keras first, then .h5 as fallback
        model_files = {
            'bilstm': ('bilstm_model.keras', 'bilstm_model_best.h5'),
            'tcn': ('tcn_model.keras', 'tcn_model_best.h5'),
            'lstm': ('lstm_model.keras', 'lstm_model_best.h5')
        }
        
        for model_name, (keras_file, h5_file) in model_files.items():
            model_loaded = False
            
            # Try .keras file first
            keras_path = f'models/{keras_file}'
            logger.info(f"Checking for {model_name} model at: {keras_path}")
            if os.path.exists(keras_path):
                try:
                    if model_name == 'tcn':
                        # TCN model needs custom objects
                        try:
                            from tcn import TCN
                            models[model_name] = load_model(keras_path, custom_objects={'TCN': TCN})
                        except ImportError:
                            logger.warning(f"TCN library not available, skipping {model_name} model")
                            continue
                    else:
                        models[model_name] = load_model(keras_path)
                    logger.info(f"Loaded {model_name} model successfully from {keras_path}")
                    model_loaded = True
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name} from .keras: {str(model_error)}")
            else:
                logger.info(f"File not found: {keras_path}")
            
            # Try .h5 file as fallback
            if not model_loaded:
                h5_path = f'models/{h5_file}'
                logger.info(f"Trying fallback {model_name} model at: {h5_path}")
                if os.path.exists(h5_path):
                    try:
                        if model_name == 'tcn':
                            try:
                                from tcn import TCN
                                models[model_name] = load_model(h5_path, custom_objects={'TCN': TCN})
                            except ImportError:
                                logger.warning(f"TCN library not available, skipping {model_name} model")
                                continue
                        else:
                            models[model_name] = load_model(h5_path)
                        logger.info(f"Loaded {model_name} model successfully from {h5_path}")
                        model_loaded = True
                    except Exception as model_error:
                        logger.error(f"Failed to load {model_name} from .h5: {str(model_error)}")
                else:
                    logger.info(f"File not found: {h5_path}")
            
            if not model_loaded:
                logger.warning(f"Could not load {model_name} model from either .keras or .h5 file")
        
        # Load scalers - try individual model scalers first
        scalers_loaded = False
        
        # Try to load from any individual model scaler file (they should all be the same)
        for model_name in ['lstm', 'bilstm', 'tcn']:
            scaler_path = f'models/{model_name}_model_scalers.pkl'
            logger.info(f"Checking for scalers at: {scaler_path}")
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as f:
                        scalers = pickle.load(f)
                    logger.info(f"Loaded scalers from {scaler_path}")
                    logger.info(f"Available scalers: {list(scalers.keys())}")
                    scalers_loaded = True
                    break
                except Exception as scaler_error:
                    logger.warning(f"Failed to load scalers from {scaler_path}: {str(scaler_error)}")
            else:
                logger.info(f"Scaler file not found: {scaler_path}")
        
        # Create default scalers if none were loaded
        if not scalers_loaded:
            logger.warning("Could not load scalers from any source, creating defaults")
            scalers = create_default_scalers()
            
    except Exception as e:
        logger.error(f"Error loading models/scalers: {str(e)}")
        models = {}
        scalers = create_default_scalers()

def create_default_scalers():
    """Create default scalers for when loading fails"""
    default_scalers = {
        'seq_voltage': MinMaxScaler(),
        'seq_current': MinMaxScaler(),
        'seq_temperature': MinMaxScaler(),
        'additional_features': StandardScaler(),
        'capacity': MinMaxScaler()
    }
    
    # Fit with some dummy data so they work
    dummy_data = np.array([[3.0], [3.5], [4.0], [4.2]])  # Realistic voltage range
    default_scalers['seq_voltage'].fit(dummy_data)
    
    dummy_current = np.array([[0.0], [0.5], [1.0], [2.0]])  # Realistic current range
    default_scalers['seq_current'].fit(dummy_current)
    
    dummy_temp = np.array([[0.0], [25.0], [35.0], [60.0]])  # Realistic temperature range
    default_scalers['seq_temperature'].fit(dummy_temp)
    
    # 4 additional features: duration_hours, voltage_drop, temperature_mean, temperature_range
    dummy_features = np.array([[1.0, 0.5, 25.0, 10.0], [2.0, 1.0, 30.0, 15.0], [3.0, 1.5, 35.0, 20.0]])
    default_scalers['additional_features'].fit(dummy_features)
    
    dummy_capacity = np.array([[1.0], [1.5], [2.0], [2.5]])  # Realistic capacity range
    default_scalers['capacity'].fit(dummy_capacity)
    
    logger.info("Created and fitted default scalers")
    return default_scalers

def normalize_sequences(voltage, current, temperature):
    """Normalize input sequences using loaded scalers"""
    try:
        # Check if scalers are properly fitted
        if not hasattr(scalers.get('seq_voltage'), 'scale_'):
            logger.warning("Scalers not fitted, using default normalization")
            
            # Simple min-max normalization with typical battery ranges
            voltage_normalized = np.array([(v - 3.0) / (4.2 - 3.0) for v in voltage])
            current_normalized = np.array([(c - 0.0) / (3.0 - 0.0) for c in current])  
            temperature_normalized = np.array([(t - 0.0) / (60.0 - 0.0) for t in temperature])
            
            return voltage_normalized, current_normalized, temperature_normalized
        
        # Normalize voltage
        voltage_array = np.array(voltage).reshape(-1, 1)
        voltage_normalized = scalers['seq_voltage'].transform(voltage_array).flatten()
        
        # Normalize current
        current_array = np.array(current).reshape(-1, 1)
        current_normalized = scalers['seq_current'].transform(current_array).flatten()
        
        # Normalize temperature
        temperature_array = np.array(temperature).reshape(-1, 1)
        temperature_normalized = scalers['seq_temperature'].transform(temperature_array).flatten()
        
        return voltage_normalized, current_normalized, temperature_normalized
    except Exception as e:
        logger.error(f"Error normalizing sequences: {str(e)}")
        # Fallback to simple normalization
        voltage_normalized = np.array([(v - 3.0) / (4.2 - 3.0) for v in voltage])
        current_normalized = np.array([(c - 0.0) / (3.0 - 0.0) for c in current])  
        temperature_normalized = np.array([(t - 0.0) / (60.0 - 0.0) for t in temperature])
        
        return voltage_normalized, current_normalized, temperature_normalized

def prepare_model_inputs(voltage, current, temperature, battery_id):
    """Prepare inputs for different models - matches your actual model architecture"""
    # Normalize sequences
    voltage_norm, current_norm, temp_norm = normalize_sequences(voltage, current, temperature)
    
    # Ensure all sequences have the same length (pad or truncate if needed)
    target_length = 100  # Based on your actual training data
    
    def pad_or_truncate(seq, target_len):
        if len(seq) > target_len:
            return seq[:target_len]
        elif len(seq) < target_len:
            return np.pad(seq, (0, target_len - len(seq)), mode='constant', constant_values=seq[-1])
        return seq
    
    voltage_norm = pad_or_truncate(voltage_norm, target_length)
    current_norm = pad_or_truncate(current_norm, target_length)
    temp_norm = pad_or_truncate(temp_norm, target_length)
    
    # Prepare sequence input (shape: [1, 100, 3])
    sequence_input = np.array([voltage_norm, current_norm, temp_norm]).T
    sequence_input = sequence_input.reshape(1, target_length, 3)
    
    # Calculate the 4 additional features that match your training data
    voltage_seq = np.array(voltage)
    current_seq = np.array(current)
    temp_seq = np.array(temperature)
    
    # Features: duration_hours, voltage_drop, temperature_mean, temperature_range
    duration_hours = len(voltage_seq) * 0.1  # Estimated duration based on sequence length
    voltage_drop = voltage_seq[0] - voltage_seq[-1] if len(voltage_seq) > 1 else 0.0
    temperature_mean = np.mean(temp_seq)
    temperature_range = np.max(temp_seq) - np.min(temp_seq) if len(temp_seq) > 1 else 0.0
    
    additional_features = np.array([
        duration_hours, voltage_drop, temperature_mean, temperature_range
    ]).reshape(1, -1)
    
    # Normalize additional features
    if 'additional_features' in scalers and hasattr(scalers['additional_features'], 'scale_'):
        try:
            additional_features = scalers['additional_features'].transform(additional_features)
        except Exception as e:
            logger.warning(f"Using fallback normalization for additional features: {str(e)}")
            # Fallback normalization based on typical ranges
            additional_features = additional_features / np.array([5.0, 1.0, 30.0, 20.0]).reshape(1, -1)
    else:
        # Simple normalization
        additional_features = additional_features / np.array([5.0, 1.0, 30.0, 20.0]).reshape(1, -1)
    
    return sequence_input, additional_features

def calculate_soh(predicted_capacity, battery_id):
    """Calculate State of Health percentage"""
    nominal_capacity = BATTERY_SPECS['nominal_capacity']
    soh_percentage = (predicted_capacity / nominal_capacity) * 100
    # Cap at 100% maximum
    soh_percentage = min(100.0, max(0.0, soh_percentage))
    return soh_percentage

def get_battery_info(battery_id):
    """Get information about the selected battery"""
    if battery_id in BATTERY_DATABASE['train_batteries']:
        dataset_type = "Training"
        index = BATTERY_DATABASE['train_batteries'].index(battery_id) + 1
    elif battery_id in BATTERY_DATABASE['test_batteries']:
        dataset_type = "Test"
        index = BATTERY_DATABASE['test_batteries'].index(battery_id) + 1
    else:
        dataset_type = "Unknown"
        index = 0
    
    return {
        "battery_id": battery_id,
        "dataset_type": dataset_type,
        "index_in_dataset": index,
        "nominal_capacity_ah": BATTERY_SPECS['nominal_capacity']
    }

@app.get("/v3/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    available_models = [name for name in ['lstm', 'bilstm', 'tcn'] if name in models]
    
    return HealthResponse(
        status="healthy",
        models_loaded={name: name in models for name in ['bilstm', 'tcn', 'lstm']},
        available_models=available_models,
        available_batteries={
            "train_batteries": BATTERY_DATABASE['train_batteries'],
            "test_batteries": BATTERY_DATABASE['test_batteries']
        },
        gpu_available=gpu_available,
        timestamp=datetime.now().isoformat()
    )

@app.get("/v3/batteries", response_model=BatteryListResponse)
async def get_batteries():
    """Get list of available batteries"""
    return BatteryListResponse(
        train_batteries=BATTERY_DATABASE['train_batteries'],
        test_batteries=BATTERY_DATABASE['test_batteries'],
        total_batteries=len(BATTERY_DATABASE['train_batteries']) + len(BATTERY_DATABASE['test_batteries'])
    )

@app.post("/v3/predict", response_model=PredictionResponse)
async def predict_soh(data: BatteryData):
    """Predict battery State of Health using selected model"""
    start_time = datetime.now()
    
    try:
        # Validate input data
        if len(data.voltage) != len(data.current) or len(data.voltage) != len(data.temperature):
            raise HTTPException(
                status_code=400, 
                detail="Voltage, current, and temperature sequences must have the same length"
            )
        
        if len(data.voltage) < 5:
            raise HTTPException(
                status_code=400,
                detail="Sequences must have at least 5 data points"
            )
        
        # Validate battery ID
        all_batteries = BATTERY_DATABASE['train_batteries'] + BATTERY_DATABASE['test_batteries']
        if data.battery_id not in all_batteries:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid battery_id. Must be one of: {all_batteries}"
            )
        
        # Validate model type
        if data.model_type not in ['lstm', 'bilstm', 'tcn']:
            raise HTTPException(
                status_code=400,
                detail="model_type must be one of: 'lstm', 'bilstm', 'tcn'"
            )
        
        # Check if requested model is available
        if data.model_type not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{data.model_type}' is not available. Available models: {list(models.keys())}"
            )
        
        # Prepare inputs - all models use both sequence and additional features
        sequence_input, additional_features = prepare_model_inputs(
            data.voltage, data.current, data.temperature, data.battery_id
        )
        
        # Make prediction with selected model
        try:
            # All models (LSTM, BiLSTM, TCN) have the same input format: [sequence_input, additional_features]
            pred_norm = models[data.model_type].predict([sequence_input, additional_features], verbose=0)
            
            # Denormalize prediction
            predicted_capacity = scalers['capacity'].inverse_transform(pred_norm.reshape(-1, 1))[0, 0]
            
            # Calculate SOH percentage
            soh_percentage = calculate_soh(predicted_capacity, data.battery_id)
            
            # Calculate confidence (simple heuristic based on input quality)
            voltage_std = np.std(data.voltage)
            temp_range = max(data.temperature) - min(data.temperature)
            confidence = min(0.95, max(0.5, 1.0 - (voltage_std * 0.1) - (temp_range * 0.01)))
            
        except Exception as model_error:
            logger.error(f"{data.model_type} prediction error: {str(model_error)}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(model_error)}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        battery_info = get_battery_info(data.battery_id)
        
        return PredictionResponse(
            soh_percentage=float(soh_percentage),
            estimated_capacity=float(predicted_capacity),
            model_used=data.model_type,
            confidence_score=float(confidence),
            processing_time=processing_time,
            battery_info=battery_info,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models and scalers on startup"""
    logger.info("Starting up Battery Capacity Prediction API v3.0")
    load_models_and_scalers()
    
    # Log GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU devices available: {len(gpus)}")
        for gpu in gpus:
            logger.info(f"GPU: {gpu}")
    else:
        logger.info("No GPU devices found, using CPU")

@app.get("/v3/")
async def root():
    """Root endpoint"""
    return {
        "message": "Battery State of Health (SOH) Prediction API v3.0",
        "description": "AI-powered battery health estimation using LSTM, BiLSTM, and TCN models",
        "endpoints": {
            "health": "/v3/health",
            "batteries": "/v3/batteries", 
            "predict": "/v3/predict",
            "documentation": "/v3/docs"
        },
        "models": ["lstm", "bilstm", "tcn"],
        "total_batteries": len(BATTERY_DATABASE['train_batteries']) + len(BATTERY_DATABASE['test_batteries'])
    }

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=5003, 
        reload=True,
        log_level="info"
    )