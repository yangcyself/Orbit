#!/bin/bash

# Check if at least one directory is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: run.sh dir1/ [dir2/ dir3/ ...]"
    exit 1
fi

# Loop through the directory arguments
for dir in "$@"; do
    # Check if the directory exists
    if [ -d "$dir" ]; then
        # Run the python command with the directory
        ./orbit.sh -p source/standalone/workflows/elevatorTaking/rollout.py --cpu --headless --checkpoints  "${dir}models/model_epoch___e__.pth"
    else
        echo "Directory $dir does not exist"
    fi
done
