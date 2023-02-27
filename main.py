"""
   This file will be responsible for the following:
   - Preprocess the data
   - Load the data into the model and train it
   - Generating embeddings based on the selected model
   - Search in the embeddings for similar audio files
   - Perform evaluation of the model using information retrieval metrics
   - Plot relevant results
   - Save the model, the embeddings and the results/plot for easy access
"""

from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import src.models.params as yamnet_params
import src.models.yamnet as yamnet_model
import src.embeddings as emb


from src.data.make_dataset import main as make_dataset

def main(data_path: str, model_path: str, results_path: str):
    """Main function of the project. This will be the entry point of the application.
    """
    # Preprocess data
    print(data_path)
    make_dataset(data_path + '/ECS50', data_path + '/processed')
    
    # Load data into model and train it (for now just load the pretrained model)
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('models/yamnet.h5')
    yamnet_classes = yamnet_model.class_names('models/yamnet_class_map.csv')
    
    print(f"Loaded model with {len(yamnet_classes)} classes")
    
    # Generate embeddings
    data_type = "normal"
    emb.embeddings(data_path + '/processed', data_type)
    
    print(f"Generated embeddings for {data_type} data. ")
    
    # Search in the embeddings for similar audio files
    
    
    # Perform evaluation of the model using information retrieval metrics
    
    # Plot relevant results
    
    # Save the model, the embeddings and the results/plot for easy access
    
if __name__ == "__main__":
    main(data_path="data", model_path="models", results_path="output")