#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Download model and configuration from a remote server.")

# Add the arguments
## Example usage run/downloadmodel.py 20230615001101 100
## It is equivalent to 
## Find 
parser.add_argument("target_directory", help="The target directory to download files to.")
parser.add_argument("epoch_number", help="The epoch number of the model to download.")
parser.add_argument("--user", required=False, default="chenyang", help="The username for the remote server.")
parser.add_argument("--host", required=False, default="euler.ethz.ch" ,help="The host address of the remote server.")
parser.add_argument("--logsprefix", required=False, default="/cluster/scratch/chenyang", help="The base path on the remote server.")
parser.add_argument("--store_logs", action="store_true", default=False, help="Whether to store the logs dir of experiment")
# Parse the command line arguments
args = parser.parse_args()



# Find the files on the remote server
try:
    find_command = f"find {args.logsprefix} -name '{args.target_directory}'"
    result = subprocess.run(["ssh", f"{args.user}@{args.host}", find_command], stdout=subprocess.PIPE, text=True, check=True)
    # List of files to download
    files_to_download = result.stdout.splitlines()


    # Ensure the target directory exists
    # Download each file using scp
    for exppath in files_to_download:

        file_path = os.path.join(exppath, "models", f"model_epoch_{args.epoch_number}.pth")
        print("downloading:", f"{args.user}@{args.host}:{file_path}")
        # Define the target path
        relative_path = os.path.relpath(file_path, args.logsprefix)
        print("to:", relative_path)
        
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(relative_path), exist_ok=True)
        
        # Execute scp command to download the file
        subprocess.run(["scp", f"{args.user}@{args.host}:{file_path}", 
            relative_path], check=True)
        if(args.store_logs):
            logs_path = os.path.join(exppath, "logs")
            relative_path = os.path.relpath(logs_path, args.logsprefix)
            subprocess.run(["scp", "-r", f"{args.user}@{args.host}:{logs_path}", 
                logs_path], check=True)

    print("Files successfully copied")

except subprocess.CalledProcessError as e:
    print("Error during remote find or scp. Check the remote details and try again.")
    print(e)

