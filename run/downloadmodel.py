#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Download model and configuration from a remote server.")

# Add the arguments
## Example usage downloadmodel.py act/20230614204409 100
## It is equivalent to 
## Find 
parser.add_argument("target_directory", help="The target directory to download files to.")
parser.add_argument("epoch_number", help="The epoch number of the model to download.")
parser.add_argument("--user", required=False, default="", help="The username for the remote server.")
parser.add_argument("--host", required=False, default="" ,help="The host address of the remote server.")
parser.add_argument("--logsprefix", required=False, default="", help="The base path on the remote server.")
# Parse the command line arguments
args = parser.parse_args()



# Find the files on the remote server
try:
    find_command = f"find {args.logsprefix} -name '{args.target_directory}'"
    print(find_command)
    result = subprocess.run(["ssh", f"{args.user}@{args.host}", find_command], stdout=subprocess.PIPE, text=True, check=True)
    # List of files to download
    files_to_download = result.stdout.splitlines()


    # Ensure the target directory exists
    # Download each file using scp
    for file_path in files_to_download:
        file_path = os.path.join(file_path, "models", f"model_epoch_{args.epoch_number}.pth")
        print("downloading:", f"{args.user}@{args.host}:{file_path}")
        os.makedirs(args.target_directory, exist_ok=True)
        # Define the target path
        relative_path = os.path.relpath(file_path, args.logsprefix)
        print("to:", relative_path)
        
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(relative_path), exist_ok=True)
        
        # Execute scp command to download the file
        subprocess.run(["scp", f"{args.user}@{args.host}:{file_path}", 
            relative_path], check=True)

    print("Files successfully copied")

except subprocess.CalledProcessError as e:
    print("Error during remote find or scp. Check the remote details and try again.")
    print(e)

