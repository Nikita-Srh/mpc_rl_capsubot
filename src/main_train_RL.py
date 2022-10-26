import datetime
import os

import cProfile
import pstats

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.capsubot_env.capsubot_env import CapsubotEnv, CapsubotEnvToPoint
from src.capsubot_env.custom_logger import SummaryWriterCallback

# HYPERPARAMS
# TODO: implement LR schedule
# TODO: play with hyperparams more
N_ENVS: int = 1
LEARNING_RATE: float = float(3.0e-4)
TIMESTEPS: float = 1e5
N_STEPS = 4096
# BATCH_SIZE =
N_EPOCHS = 10
# GAMMA =
# GAE_LAMBDA =


def main(max_speed_ver: bool):
    additional_info_str = f"n_envs_{N_ENVS}_LR_{str(LEARNING_RATE)[2:]}_Nsteps_{N_STEPS}_Nepochs_{N_EPOCHS}" + datetime.datetime.now().strftime(
        "%d_%m_%Y-%H"
    )

    if max_speed_ver:
        ver = "PPO-"
        sub_folder = "max_speed"
        env = CapsubotEnv()
    else:
        ver = "TO_POINT_PPO-"
        sub_folder = "to_point"
        env = CapsubotEnvToPoint()

    models_dir: str = os.path.join(
        "RL_WIP", "RL_data_store", "models", sub_folder, ver
    ) + additional_info_str

    logdir: str = os.path.join(
        "RL_WIP", "RL_data_store", "logs", sub_folder, ver
    ) + additional_info_str

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = make_vec_env(lambda: env, n_envs=N_ENVS)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        tensorboard_log=logdir,
        n_steps=N_STEPS,
        n_epochs=N_EPOCHS,
    )

    # pr = cProfile.Profile()

    for i in range(1, 70):
        # pr.enable()
        model.learn(
            total_timesteps=int(TIMESTEPS),
            reset_num_timesteps=False,
            tb_log_name="PPO_xGrT0_3andxSmT-0_08",
            callback=SummaryWriterCallback(),
        )
        # pr.disable()
        model.save(os.path.join(models_dir, f"{int(TIMESTEPS * i)}"))
        # ps = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        # ps.print_stats(50)


if __name__ == "__main__":
    main(max_speed_ver=False)
