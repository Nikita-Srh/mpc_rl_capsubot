import os

import numpy as np
from stable_baselines3 import PPO

from src.capsubot_env.capsubot_env import CapsubotEnv

from src.capsubot_env.capsubot_env_to_point import CapsubotEnvToPoint

# insert into model_path the path to the model *.zip
# it can't be hardcoded because of using datetime module
models_dir: str = os.path.join("..", "RL_WIP", "RL_data_store", "models")
model_path: str = os.path.join(
    models_dir,
    # "to_point",
    "PPO-n_envs_1_LR_00025_nsteps_409613_09_2022-04",
    "2000000",
)


def printer(array: list) -> None:
    array = np.array(array)
    print(f"min_value = {np.amin(array)}, max_value = {np.amax(array)}")
    print("-----------------------------------------------------------")


model = PPO.load(model_path)

env = CapsubotEnv(is_render=True)
obs = env.reset()
n_steps = int(5.0 / env.dt)
rewards = []
xs = []
x_dots = []
xis = []
xi_dots = []
actions = [0]
states = {"x": xs, "x_dot": x_dots, "xi": xis, "xi_dot": xi_dots}
for step in range(2000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    xs.append(env.agent_state[0])
    x_dots.append(env.agent_state[1])
    xis.append(env.agent_state[2])
    xi_dots.append(env.agent_state[3])

    rewards.append(reward)
    actions.append(action)
    env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print(
            f"Goal reached! reward={reward}, "
            f"at time={info.get('total_time')}, "
            f"x_pos = {info.get('obs_state').get('agent')[0]}, "
            f"average_speed = {info.get('average_speed')}"
        )

        print("x is ")
        printer(states.get("x"))
        print("x_dot is ")
        printer(states.get("x_dot"))
        print("xi is ")
        printer(states.get("xi"))
        print("xi_dot is ")
        printer(states.get("xi_dot"))

        break
env.close()
