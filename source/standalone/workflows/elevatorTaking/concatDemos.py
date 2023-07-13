import h5py
import os
import shutil
import argparse
from datetime import datetime

# Define a recursive function to copy group structure and create virtual datasets.
def copy_structure(source_group, target_group):
    for name, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            vsource = h5py.VirtualSource(item)
            layout = h5py.VirtualLayout(shape=item.shape, dtype=item.dtype)
            layout[:] = vsource
            target_group.create_virtual_dataset(name, layout)
        elif isinstance(item, h5py.Group):
            target_subgroup = target_group.create_group(name)
            copy_structure(item, target_subgroup)

# Use this function to create a new file from existing files.
def create_dataset_stack(source_files, target_file):
    with h5py.File(target_file, 'w') as f_target:
        data_group = f_target.create_group("data")
        data_group.attrs["total"] = 0
        demo_count = 0
        for source_file in source_files:
            with h5py.File(source_file, 'r') as f_source:
                if (data_group.attrs["total"] == 0):
                    data_group.attrs["env_args"] = f_source["data"].attrs["env_args"]
                
                data_group.attrs["total"] += f_source["data"].attrs["total"]
                for demo_id in f_source["data"].keys():
                    demo_target = data_group.create_group(f"demo_{demo_count}")
                    demo_source = f_source["data"][demo_id]
                    copy_structure(demo_source, demo_target)
                    demo_count += 1


def create_dataset_dir(sources, target_dir):
    os.makedirs(target_dir , exist_ok=True)
    for source in sources:
        basename = os.path.basename(source)
        shutil.copytree(os.path.join(source, "params"), os.path.join(target_dir, basename+"_params"))
    with open(os.path.join(target_dir, "sources.txt"), "w") as f:
        f.write("\n".join(sources))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Add paths (positional argument)
    parser.add_argument('dataset_paths', nargs='+', help="List of paths to hdf5 dataset")
    parser.add_argument("--prefix", type=str, default=".", help="Save dataset to <prefix>/logs/rolloutCollection.")
    parser.add_argument("--task", type=str, default="pushbtn", help="Name of the task: can be pushbtn or movetobtn")
    args = parser.parse_args()
    sources = args.dataset_paths

    log_dir = datetime.now().strftime("G%b%d_%H-%M-%S")
    log_dir = os.path.join(args.prefix, "logs/rolloutCollection", f"Elevator-{args.task}", log_dir)

    create_dataset_dir([os.path.dirname(s) for s in sources], log_dir)
    create_dataset_stack(sources, os.path.join(log_dir, "hdf_dataset.hdf5"))

