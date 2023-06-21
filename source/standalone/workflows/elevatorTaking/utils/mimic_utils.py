import gym
import torch
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_gym import EnvGym
import robomimic.utils.file_utils as FileUtils

class RobomimicWrapper(RolloutPolicy):
    """The Wrapper of the RolloffPolicy
    """
    def __init__(self, checkpoint, device, verbose=False):
        self.device = device
        self.policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint, device=device, verbose=verbose)

    def start_episode(self):
        self.policy.start_episode()
    
    def _prepare_observation(self, ob):
        self.policy._prepare_observation(ob)

    def __call__(self, ob, goal=None):
        if "goal" in ob:
            ob["goal"] = {k[5:]:v for k,v in ob["goal"].items()} # remove the "goal_" prefix
        obs = {f"{kk}:{k}":v[0] for kk,vv in ob.items() for k,v in vv.items()}
        obs["rgb:hand_camera_rgb"] = obs["rgb:hand_camera_rgb"].permute(2, 0, 1).to(dtype=torch.float32)/255.
        obs["rgb:hand_camera_rgb"] = obs["rgb:hand_camera_rgb"].clamp(min =0., max=1.)
        if "goal" in ob:
            obs["goal:rgb"] = obs["goal:rgb"].permute(2, 0, 1).to(dtype=torch.float32)/255.
            obs["goal:semantic"] = obs["goal:semantic"].permute(2, 0, 1).to(dtype=torch.float32)/255.
        return torch.tensor(self.policy(obs)).to(self.device)[None,...]

class myEnvGym(EnvGym):

    def __init__(self, *args, **kwargs):
        super(myEnvGym, self).__init__(*args, **kwargs)
        self._bare_env = self.env
    
    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.
        """
        return obs if obs is not None else self._current_obs
    
    def reset(self):
        obs = self.env.reset()
        self._current_obs = obs
        return self.get_observation(obs)
    

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, reward, done, info = self.env.step(action)
        self._current_obs = obs
        self._current_reward = reward
        self._current_done = done
        return self.get_observation(obs), reward.detach().cpu().numpy(), self.is_done(), info

    def get_state(self):
        return self.env.get_state()
    
    def reset_to(self, state):
        return self.env.reset_to(state)

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        return {
            k : v.cpu().numpy() 
            for k,v in self.env.is_success().items()
        }

    def close(self):
        self.env.close()

    def wrap_with_gym_recorder(self, video_kwargs):
        self.env = gym.wrappers.RecordVideo(self._bare_env, **video_kwargs)
