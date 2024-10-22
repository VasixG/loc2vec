#!/bin/bash

echo "Starting preprocessing..."
python preprocessing.py --config preprocessing_config.yaml

if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully."
    
    echo "Starting training..."
    python train.py --config training_config.yaml
    
    if [ $? -eq 0 ]; then
        echo "Training completed successfully."
    else
        echo "Training failed."
        exit 1
    fi
else
    echo "Preprocessing failed."
    exit 1
fi
