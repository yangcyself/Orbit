#!/bin/bash

# Check if at least two arguments are provided (option and directory)
if [ "$#" -lt 2 ]; then
    echo "Usage: rollout.sh [-c1 | -r] dir1 [-c2 | -r] dir2 ..."
    echo "-c1: start a new rollout: --checkpoint with --task=pushbtn"
    echo "  dirs: directory of the experiment"
    echo "-c2: start a new rollout: --checkpoint with --task=movetobtn"
    echo "  dirs: directory of the experiment"
    echo "-r: resume previous: --resume"
    echo "  dirs: directory of the rollout"
    exit 1
fi

orig_command="$0 $*"
start_time=$(date)

# Parse the arguments
while [ "$#" -gt 0 ]; do
    case $1 in
        -c1)
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
                ./orbit.sh -p source/standalone/workflows/elevatorTaking/rollout.py --cpu --headless --task pushbtn --checkpoints "${dir}models/model_epoch___e__.pth"
            else
                echo "Directory $dir does not exist"
            fi
            shift # Move to next argument
            ;;
        -c2)
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
                ./orbit.sh -p source/standalone/workflows/elevatorTaking/rollout.py --cpu --headless --task movetobtn --checkpoints "${dir}models/model_epoch___e__.pth"
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
            echo "Usage: run.sh [-c1 | -r] dir1 [-c2 | -r] dir2 ..."
            exit 1
            ;;
    esac
done

sendMeTelegram.sh "Rollout finished" $'\n' "${orig_command}" $'\n\n' "start: ${start_time}" $'\n' "end:   $(date)"
