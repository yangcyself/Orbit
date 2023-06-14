## Debug process with demoCollection and playback_dataset

Run democollection with `--debug`

```bash
./orbit.sh -p source/standalone/workflows/elevatorTaking/demoCollection.py --cpu --num_demos=5  --checkpoint logs/rsl_rl/elevator_reach/Jun01_12-56-21_Good/model_700.pt --prefix /media/chenyu/T7/temp/ --debug
```

Split the dataset

```bash
./orbit.sh -p source/standalone//workflows/robomimic/tools/split_train_val.py /media/chenyu/T7/temp/logs/rolloutCollection/Isaac-Elevator-Franka-v0/Jun14_11-49-56/hdf_dataset.hdf5 --ratio 0.2
```

Playback dataset with `--debug` flag

```bash
./orbit.sh -p source/standalone/workflows/elevatorTaking/playback_dataset.py --use-actions --dataset /media/chenyu/T7/temp/logs/rolloutCollection/Isaac-Elevator-Franka-v0/Jun14_11-49-56/hdf_dataset.hdf5 --cfg /home/chenyu/opt/robomimic/robomimic/workflow/cfgs/iad.json --video_path ~/falseHeadless.mp4 --render_image_names "rgb:hand_camera_rgb" --n 10 --headless --debug
```

> Note that I can play with the config of the dataloader. Such as tuning the flag `whole_traj_seq` and sequence length

> Now I get some warnings for the errors between the state in simulation and the state in playback at a scale of 1e-4

Then check the generated video, it should be identical on both sides

