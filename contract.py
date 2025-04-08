from typing import Any, Dict, List
from pydantic import BaseModel

class ImagePreprocessRequest(BaseModel):
    dataset_path: str
    options: Dict[str, Any] = {
        "resize": (28,28),
        "rgb2gray": 0,
        "shuffle": 0,
        "normalize": 0,
        "save_option": "one image per class"
    }

class TrainingRequest(BaseModel):
    dataset_path: str
    training_options: Dict[str, Any] = {
        "method": "train_test_val",
        "layers": [{"type": "Flatten"},  {"type": "Dense", "units": 512, "activation": "relu"}, {"type": "Dense", "units": 10, "activation": "softmax"}],
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "lr": 0.01,
        "epochs": 10,
        "batch_size": 64,
        "kFold_k": 5
    }

class PredictionRequest(BaseModel):
    model_path: str
    input_data: str
    preprocessing_options: Dict[str, Any] = {
        "resize": (28,28),
        "rgb2gray": 0,
        "save_option": "one image per class"
    }

class TestingRequest(BaseModel):
    model_path: str
    dataset_path: str
    preprocessing_options: Dict[str, Any] = {
        "resize": (28,28),
        "rgb2gray": 0,
        "save_option": "one image per class"
    }