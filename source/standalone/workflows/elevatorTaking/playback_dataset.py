"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

from omni.isaac.kit import SimulationApp
config = {"headless": False}
simulation_app = SimulationApp(config)

import os
import sys
import json
import h5py
import argparse
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.config import config_factory
import json
from robomimic.envs.env_base import EnvBase, EnvType
import time

import gym
import torch
from torch.utils.data import DataLoader
import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg

sys.path.append(os.path.dirname(__file__))
from utils.mimic_utils import RobomimicWrapper, myEnvGym


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def playback_trajectory_with_env(
    env, 
    states, 
    obs_dict = None,
    actions=None, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        states (torch.tensor): seq, dim:  array of simulation states to load
        actions (torch.tensor): if provided, play actions back open-loop instead of using @states
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """

    write_video = (video_writer is not None)
    video_count = 0

    # load the initial state
    env.reset()
    env.reset_to(states[0].unsqueeze(0))

    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    if(camera_names is not None):
        video_count = 0

    for i in range(traj_len):
        if action_playback:
            env.step(actions[i].unsqueeze(0))
            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = env.get_state()
                # print("states[i+1]", states[i+1])
                # print("state_playback",state_playback)
                err = torch.norm(states[i + 1] - state_playback.squeeze(0))
                if err > 1e-3:
                    print("warning: playback diverged by {} at step {}".format(err, i))
                    # print("states:", states[i + 1])
                    # print("state_playback", state_playback)
            if env.is_success()["task"]:
                print("TASK SUCCESS")
            image_names = ["rgb:hand_camera_rgb"]
            if(camera_names is not None): # compare the image from the simulator with the image in the dataset
                if video_count % video_skip == 0:
                    # concatenate image obs together
                    im_playback = [obs_dict[k][i] for k in image_names]
                    obs = env.get_observation()
                    im_sim = [obs[k.split(":")[0]][k.split(":")[1]][0].permute(2, 0, 1).to(dtype=torch.float32)/256.
                        for k in image_names]
                    frame_playback = torch.cat(im_playback, axis=1)
                    frame_sim = torch.cat(im_sim, axis=1)
                    frame = torch.cat([frame_playback, frame_sim], axis=2)
                    video_writer.append_data(frame.permute(1,2,0).numpy())
                video_count += 1

        else:
            raise ValueError("Must playback with action, add `--use-actions` in the cmd")
            env.reset_to(states[i].unsqueeze(0))


        if first:
            break


def playback_trajectory_with_obs(
    traj_grp,
    video_writer, 
    video_skip=5, 
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def get_iterator_from_dataset(args):
    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]
    
    for ind in range(len(demos)):
        ep = demos[ind]
        print("Loading episode: {}".format(ep))
        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]

        traj_grp = f["data/{}".format(ep)]

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
        yield {k:torch.tensor(traj_grp[f"obs/{k}"]) for k in traj_grp["obs"].keys()}, torch.tensor(states), torch.tensor(actions)
    f.close()


def get_iterator_from_dataloader(args):
    config = config_factory("ycy")
    ext_cfg = json.load(open(args.cfg, 'r'))
    with config.values_unlocked():
        config.update(ext_cfg)
    config.train.data = args.dataset
    config.train.batch_size = 1
    config.train.dataset_keys = ["states", "actions"]
    ObsUtils.initialize_obs_utils_with_config(config)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=args.dataset,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )
        
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    train_loader = DataLoader(
        dataset=trainset,
        sampler=None, # turn off shuffle
        batch_size=config.train.batch_size,
        shuffle=None,
        num_workers=config.train.num_data_workers,
        drop_last=True
    )
    train_loader_iter = iter(train_loader)
    for i, batch in enumerate(train_loader_iter):
        if(args.n is not None and i>args.n):
            break
        states = batch["states"]
        actions = batch["actions"]
        actions[~batch["pad_mask"].squeeze(2),:]=0
        yield {k:v.squeeze(0) for k,v in batch["obs"].items()}, states.squeeze(0), actions.squeeze(0)


def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        # env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        # env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)
        print("making env!!")
        env_cfg = parse_env_cfg("Isaac-Elevator-Franka-v0", use_gpu=False, num_envs=1)
        env_cfg.env.episode_length_s = 20.
        env_cfg.initialization.elevator.max_init_floor = 5 # wait for at most 5 seconds
        env_cfg.initialization.elevator.moving_elevator_prob = 0 # wait for at most 5 seconds
        env_cfg.initialization.elevator.nonzero_floor_prob = 1 # wait for at most 5 seconds
        env_cfg.initialization.robot.position_uniform_min = [1.4, 0.9, -1.6]  # position (x,y,z)
        env_cfg.initialization.robot.position_uniform_max = [1.6, 1.1, -1.4]  # position (x,y,z)
        
        env_cfg.terminations.episode_timeout = True
        env_cfg.terminations.is_success = "pushed_btn"
        env_cfg.terminations.collision = True
        env_cfg.observations.return_dict_obs_in_group = True
        env_cfg.observation_grouping = {"policy":"privilege", "rgb":None}

        # env = gym.make("Isaac-Elevator-Franka-v0", cfg=env_cfg, headless=False)
        env = myEnvGym("Isaac-Elevator-Franka-v0", cfg=env_cfg, headless=True)
        print("env_made!!")

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    if args.cfg is None:
        dataset_iter = get_iterator_from_dataset(args)
    else:
        dataset_iter = get_iterator_from_dataloader(args)

    for obs_dict, states, actions in dataset_iter:
        if args.use_obs:
            playback_trajectory_with_obs(
                obs_dict=obs_dict, 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue
        else:
            playback_trajectory_with_env(
                env=env, 
                obs_dict=obs_dict,
                states=states, actions=actions, 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                camera_names=args.render_image_names,
                first=args.first,
            )

    if write_video:
        video_writer.close()
    # close the simulator
    if not args.use_obs:
        env.close()
    simulation_app.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="path to config file. If not none, will load trajectories from dataloader"
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    args = parser.parse_args()
    playback_dataset(args)
