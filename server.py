import logging
from fastapi import FastAPI, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from typing import Tuple
import os
import uuid
import base64
import pandas as pd

import ai
import contract

DATASETS_DIR = "datasets"
CHECKPOINTS_DIR = "checkpoints"

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
enable_backtrace = os.environ.get("ENABLE_BACKTRACE", "0") == "1"

app = FastAPI(default_response_class=ORJSONResponse)

@app.middleware("http")
async def handle_errors(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        if enable_backtrace:
            logging.error(f"Error processing request: {e}")
        
        return ORJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload CSV file and convert it to HDF5 format.
    """
    # Verify file type
    if file.content_type != "text/csv":
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Only CSV files are accepted"}
        )
        
    # Get original filename and create .h5 filename
    original_filename = file.filename
    base_name = os.path.splitext(original_filename)[0]
    h5_filename = f"{base_name}.h5"
    
    # Check if filename already exists
    if os.path.exists(DATASETS_DIR + "/" + h5_filename):
        return ORJSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": f"A file with name '{h5_filename}' already exists. Please use a different filename."}
        )
    
    # Read file content
    file_content = await file.read()
    file_content_str = file_content.decode('utf-8')

    def load_csv(file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process CSV file to extract images and labels
        """
        decode = lambda x: np.array(Image.open(BytesIO(base64.b64decode(x))))

        # read the uploaded file
        data = StringIO(file)
        df = pd.read_csv(data)
        images = df['image']
        images = np.array(list(map(decode, images)))
        labels = df['label']
        labels = np.array(labels)

        return images, labels
    
    # Process CSV data
    images, labels = load_csv(file_content_str)

    # Save as a dataset
    dataset = ai.Dataset(images=images, labels=labels)
    dataset.save_dataset(DATASETS_DIR + "/" + h5_filename)

    return h5_filename

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    """
    Delete a file from the server.
    """
    # Prevent directory traversal
    if "/" in filename or "\\" in filename or ".." in filename or ".." in filename or "~" in filename:
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid filename"}
        )

    filename = DATASETS_DIR + "/" + filename

    # Check if file exists
    if not os.path.exists(filename) or not filename.endswith('.h5'):
        return ORJSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": f"File '{filename}' not found or not a valid H5 file"}
        )

    # Delete the file
    os.remove(filename)
    return ORJSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": f"File '{filename}' successfully deleted"}
    )

@app.post("/preprocess")
async def preprocess(request: contract.ImagePreprocessRequest):
    """
    Preprocess the dataset based on the options provided in the request.
    """
    options = request.options

    preprocessing = ai.Preprocessing(filename=DATASETS_DIR + "/" + request.dataset_path)
    output_paths = {}

    # Set save option
    save_option = "None" # by default

    # Preprocess the dataset based on the options provided
    for option in options:
        if option=="resize":
            width, height = options["resize"]
            preprocessing.resize(width, height)

        if option=="grayscale":
            preprocessing.convert_to_grayscale()

        if option=="normalize":
            preprocessing.normalize()

        if option=="shuffle":
            preprocessing.shuffle_data()

        if option=="resize" or option=="grayscale":
            if save_option == "whole dataset":
                output_path = option + os.path.basename(request.dataset_path)
                preprocessing.save_dataset(DATASETS_DIR + "/" + output_path)
                output_paths[option] = output_path
            elif save_option == "one image per class":
                output_paths[option] = preprocessing.return_class_example()

    output_path = f"preprocessed_{os.path.basename(request.dataset_path)}"
    preprocessing.save_dataset(DATASETS_DIR + "/" + output_path)
    output_paths["output"] = output_path

    # X = preprocessing.get_x()
    # y = preprocessing.get_y()
    # df = pd.DataFrame(X.reshape(X.shape[0], -1))
    # df["label"] = y
    # df.to_csv('data.csv', index=False)

    return output_paths
        
@app.post("/train")
async def train(request: contract.TrainingRequest):
    """
    Process training request, return model path and performance graphs.
    """    
    # Load dataset
    dataset = ai.Dataset()
    X, y = dataset.load_saved_dataset(DATASETS_DIR + "/" + request.dataset_path)

    # Preprocess the dataset based on the options provided
    cnn = ai.CNN(X, y, request.training_options)
    cnn.train_model()
    
    # Save model
    model_path = f"model_{uuid.uuid4().hex[:8]}.keras"
    cnn.save_model(CHECKPOINTS_DIR + "/" + model_path)

    loss_graph, accuracy_graph = cnn.get_performance_graphs()
    loss_data, accuracy_data = cnn.get_performance_data()

    # Convert DataFrames to dict with lists format to match TypeScript interface
    accuracy_data_dict = accuracy_data.to_dict(orient='list')
    loss_data_dict = loss_data.to_dict(orient='list')

    # WHY IS IT LIKE THIS NOW
    return {"model path": model_path, "accuracy graph": accuracy_graph, "loss graph": loss_graph,
            "accuracy data": accuracy_data_dict, "loss data": loss_data_dict}

# after trainig the model, use it to predict a set of images
@app.post("/predict")
async def predict(request: contract.PredictionRequest):
    """
    Process prediction request, return predicted class and confidence level.
    """
    image_data = base64.b64decode(request.input_data)
    image = Image.open(BytesIO(image_data))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_array = np.array(image)

    preprocessing = ai.Preprocessing(X=[image_array], y=[0])
    options = request.preprocessing_options

    save_option = "none"

    output_paths = {}
    for option in options:
        if option=="resize":
            width, height = options["resize"]
            preprocessing.resize(width, height)

        if option=="grayscale":
            preprocessing.convert_to_grayscale()

        if option=="resize" or option=="grayscale":
            if save_option == "whole dataset":
                output_path = option + os.path.basename(uuid.uuid4().hex[:8])
                preprocessing.save_dataset(output_path)
                output_paths[option] = output_path
            elif save_option == "one image per class":
                output_paths[option] = preprocessing.return_class_example()

    image = np.array(preprocessing.get_x())[0]
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    cnn = ai.CNN()
    cnn.load_model(CHECKPOINTS_DIR + "/" + request.model_path)
    prediction, confidence = cnn.run_model(image)

    # Convert ndarray to list for JSON serialization
    prediction = prediction.tolist()
    confidence = confidence.tolist()
    
    return {"predicted class": prediction, "confidence level": confidence, "intermediates": output_paths}
    
@app.post("/test")
async def test(request: contract.TestingRequest):
    """
    Process testing request, return accuracy and performance graphs.
    """
    preprocessing = ai.Preprocessing(filename=DATASETS_DIR + "/" + request.dataset_path)
    options = request.preprocessing_options
    output_paths = {}
    labels = np.array(preprocessing.get_y())

    save_option = "none"

    for option in options:
        if option=="resize":
            width, height = options["resize"]
            preprocessing.resize(width, height)

        if option=="grayscale":
            preprocessing.convert_to_grayscale()

        if option=="resize" or option=="grayscale":
            if save_option == "whole dataset":
                output_path = option + os.path.basename(uuid.uuid4().hex[:8])
                preprocessing.save_dataset(output_path)
                output_paths[option] = output_path
            elif save_option == "one image per class":
                output_paths[option] = preprocessing.return_class_example()

    images = np.array(preprocessing.get_x())
    labels = np.array(preprocessing.get_y())

    cnn = ai.CNN()
    cnn.load_model(CHECKPOINTS_DIR + "/" + request.model_path)
    accuracy, accuracy_per_class = cnn.test_model(images, labels)

    return {"accuracy": accuracy, "accuracy per class": accuracy_per_class, "intermediates": output_paths}


if __name__ == "__main__":
    # Create datasets folder if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
