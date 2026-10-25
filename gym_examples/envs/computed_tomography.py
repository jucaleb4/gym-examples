import gymnasium as gym
from gymnasium import spaces

import numpy as np
import numpy.linalg as la
from skimage.transform import radon, iradon_sart

from . import PhantomGenerator, utils
from .PhantomGenerator import PhantomGenerator
from .utils import ImageType, RewardType

def angle_range(N_a):
    return np.linspace(0,np.pi,N_a,False)

def forward_eval(image, theta, percentage=0.0):
    sinogram = radon(image, theta=theta) # circle=False
    n = np.random.normal(0, np.std(sinogram), (len(sinogram), len(theta))) * percentage
    sinogram_n = sinogram + n

    return sinogram_n

def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    #print(rmse)
    return 20*math.log10(1.0/rmse)

class ComputedTomographyEnv(gym.Env):
    def __init__(self, seed, n_images, num_angles, image_size, action_size,
        corruption_percentage, pnsr_reg, image_type, reward_type,
        angle_in_degrees=False,
    ):
        # Generate phantoms
        gen = PhantomGenerator(seed=seed, image_size=image_size)

        if image_type.value == ImageType.MIXED.value:
            self.P_all = gen.generate_mixed(n_samples=n_images)
        elif image_type.value == ImageType.TRIANGLE.value:
            self.P_all = gen.generate_triangle(n_samples=n_images)
        elif image_type.value == ImageType.TRIANGLE_THIRTY.value:
            self.P_all = gen.generate_triangle(n_samples=n_images, max_angle=30)
        elif image_type.value == ImageType.TRIANGLE_FIXED.value:
            self.P_all = gen.generate_triangle(n_samples=n_images, max_angle=0)
        elif image_type.value == ImageType.LINE.value:
            self.P_all = gen.generate_line(n_samples=n_images)
        elif image_type.value == ImageType.LINE_THIRTY.value:
            self.P_all = gen.generate_line(n_samples=n_images, max_angle=30)
        elif image_type.value == ImageType.LINE_FIXED.value:
            self.P_all = gen.generate_line(n_samples=n_images, max_angle=0)
        elif image_type.value == ImageType.SKIMAGE.value:
            self.P_all = gen.generate_skimage(n_samples=n_images)
        else:
            raise Exception("Unknown ImageType %s" % image_type)

        # Select phantom index
        self.n = self.np_random.choice(len(self.P_all))
        self.num_angles = num_angles
        self.reward_type = reward_type
        self.image_size = image_size
        self.action_size = action_size
        self.proj_size = int(1.5*self.image_size)
        self.vol_geom = None 
        self.angles = angle_range(self.action_size)
        if angle_in_degrees:
            self.angles = np.rad2deg(self.angles)
        self.n_iter_sirt = 150
        self.init_start = 0
        self.first_step = True
        self.corruption_percentage = max(0, corruption_percentage)
        self.pnsr_reg = pnsr_reg
        self.img_channels = 1

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.image_size, self.image_size), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_angles)

    def _get_obs(self):
        return np.array(self.state)

    def _get_info(self):
        return {"image_id": self.n}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.n = self.np_random.choice(len(self.P_all))
        self.a_start = self.init_start
        self.curr_iteration = 0
        self.angles_seq = []

        # initialization for action
        self.previous_reward = 0
        self.reward = 0
        self.done=False

        # set a zero matrix as the first reconstruction or belief state
        self.state = np.zeros((self.image_size, self.image_size))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.a_start += 1
        self.angle_action = self.angles[action]
        self.angles_seq.append(self.angle_action)

        sinogram_n = forward_eval(
            self.P_all[self.n].astype('float'),
            theta=self.angles_seq,
            percentage=self.corruption_percentage,
        )
        self.state = iradon_sart(sinogram_n, theta=self.angles_seq) # re-constructed image
        self.reward = self._get_reward(sinogram_n)

        if self.a_start > self.num_angles:
            self.a_start = self.init_start
            self.angles_seq = []
            self.done = True

        # return np.array(self.state), self.reward, self.done, self.angle_action, self.angles_seq, self.n

        observation = self._get_obs()
        info = self._get_info()

        return observation, self.reward, self.done, False, info

    def render(self):
        pass

    def _get_reward(self, sinogram_n):
        reward = 0

        if self.reward_type.value in [RewardType.PNSR_E2E.value, RewardType.PNSR_INCREMENTAL.value, RewardType.PNSR_ADD.value]:
            current_reward = psnr(self.P_all[self.n], self.state)   
            if (self.reward_type.value == RewardType.PNSR_E2E.value) and (self.a_start > self.num_angles):
                reward = current_reward
            # if PNSR_E2E and not reached num_angles, reward is sparse at 0
            elif self.reward_type.value == RewardType.PNSR_INCREMENTAL.value:
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward
            elif self.reward_type.value == RewardType.PNSR_ADD.value:
                reward = current_reward/float(self.num_angles)
        elif self.reward_type.value in [RewardType.FWD_E2E.value, RewardType.FWD_INCREMENTAL.value]:
            bootstrap_image = radon(self.state, theta=self.angles_seq)
            reconstruction_error = la.norm(sinogram_n - bootstrap_image, ord='fro')
            current_reward = -reconstruction_error/la.norm(sinogram_n, ord='fro')

            if (self.reward_type.value == RewardType.FWD_E2E.value) and (self.a_start > self.num_angles):
                reward = current_reward
            # if FWD_E2E and not reached num_angles, reward is sparse at 0
            elif self.reward_type.value == RewardType.FWD_INCREMENTAL.value:
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward
        elif self.reward_type.value in [
                RewardType.FWD_RND_1.value, RewardType.FWD_RND_10.value, 
                RewardType.FWD_RND_COMPLETE_10.value, RewardType.FWD_RND_COMPLETE_FULL.value,
                RewardType.FWD_INC_RND_1.value, RewardType.FWD_INC_RND_10.value,
        ]:
            if (self.reward_type.value in [
                    RewardType.FWD_RND_COMPLETE_10.value, RewardType.FWD_RND_COMPLETE_FULL.value,
            ]) and (self.a_start < self.num_angles):
                # has not complete
                return reward

            # TODO: Use python pattern matching 
            # https://stackoverflow.com/questions/11479816/what-is-the-python-equivalent-for-a-case-switch-statement
            n_scans = 1 if self.reward_type.value == RewardType.FWD_RND_1.value else 10
            rnd_angles_seq = self.np_random.choice(self.angles, replace=False, size=n_scans)
            if self.reward_type.value == RewardType.FWD_RND_COMPLETE_FULL.value:
                rnd_angles_seq = self.angles

            rnd_sinogram_n = forward_eval(
                self.P_all[self.n].astype('float'), 
                theta=rnd_angles_seq,
                percentage=self.corruption_percentage,
            )
            bootstrap_image = radon(self.state, theta=rnd_angles_seq)
            reconstruction_error = la.norm(rnd_sinogram_n - bootstrap_image, ord='fro')
            current_reward = -reconstruction_error/la.norm(sinogram_n, ord='fro')

            # normalize rewards
            if self.reward_type.value in [
                    RewardType.FWD_RND_1.value, RewardType.FWD_RND_10.value, 
            ]:
                # multiply by the number of scans for a fair comparison
                reward = current_reward/float(self.num_angles)
            elif self.reward_type.value in [
                    RewardType.FWD_INC_RND_1.value, RewardType.FWD_INC_RND_10.value
            ]:
                reward = current_reward - self.previous_reward
                self.previous_reward = current_reward
            else:
                reward = current_reward

        elif self.reward_type.value == RewardType.FWD_RND_1_REG.value:
            n_scans = 1 
            rnd_angles_seq = self.np_random.choice(self.angles, replace=False, size=n_scans)

            rnd_sinogram_n = forward_eval(
                self.P_all[self.n].astype('float'), 
                theta=rnd_angles_seq,
                percentage=self.corruption_percentage,
            )

            bootstrap_image = radon(self.state, theta=rnd_angles_seq)
            reconstruction_error = la.norm(rnd_sinogram_n - bootstrap_image, ord='fro')
            reconstruction_reward = -reconstruction_error/la.norm(sinogram_n, ord='fro')

            pnsr_reward = psnr(self.P_all[self.n], self.state)   

            reward = reconstruction_reward + self.pnsr_reg*pnsr_reward
            # normalize for a fair comparison
            reward /= float(self.num_angles)

        else:
            raise Exception("Unknown reward_type %s" % self.reward_type)

        return reward
