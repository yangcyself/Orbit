#!/bin/bash

# Check if at least two arguments are provided (option and directory)
if [ "$#" -lt 2 ]; then
    echo "Usage: rollout.sh [-c | -r] dir1 [-c | -r] dir2 ..."
    echo "-c: start a new rollout: --checkpoint"
    echo "  dirs: directory of the experiment"
    echo "-r: resume previous: --resume"
    echo "  dirs: directory of the rollout"
    exit 1
fi

# Parse the arguments
while [ "$#" -gt 0 ]; do
    case $1 in
        -c)
            shift # Remove the -c argument
            # Check if directory is provided after -c
            if [ "$#" -eq 0 ]; then
                echo "Error: No directory specified after -c"
                exit 1
            fi
            dir=$1
            # Check if the directory exists
            if [ -d "$dir" ]; then
                # Run the python foo command with the directory
                ./orbit.sh -p source/standalone/workflows/elevatorTaking/rollout.py --cpu --headless --checkpoints "${dir}models/model_epoch___e__.pth"
            else
                echo "Directory $dir does not exist"
            fi
            shift # Move to next argument
            ;;
        -r)
            shift # Remove the -r argument
            # Check if directory is provided after -r
            if [ "$#" -eq 0 ]; then
                echo "Error: No directory specified after -r"
                exit 1
            fi
            dir=$1
            # Check if the directory exists
            if [ -d "$dir" ]; then
                # Run the python bar command with the directory
                ./orbit.sh -p source/standalone/workflows/elevatorTaking/rollout.py --cpu --headless --resume "${dir}rollout_log.txt"
            else
                echo "Directory $dir does not exist"
            fi
            shift # Move to next argument
            ;;
        *)
            # Invalid option
            echo "Invalid option: $1"
            echo "Usage: run.sh [-c | -r] dir1 [-c | -r] dir2 ..."
            exit 1
            ;;
    esac
done