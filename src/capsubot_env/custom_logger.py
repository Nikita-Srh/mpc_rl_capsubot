import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class SummaryWriterCallback(BaseCallback):
    def _on_training_start(self) -> None:
        self._log_freq = 100  # log every 100 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            list_of_av_speeds = [
                item.get("average_speed") for item in self.locals.get("infos")
            ]
            list_of_x_pos = [
                item.get("current_pos") for item in self.locals.get("infos")
            ]
            list_of_x_velocities = [
                item.get("center_mass_velocity") for item in self.locals.get("infos")
            ]
            average_speed = np.array(sum(list_of_av_speeds) / len(list_of_av_speeds))

            self.tb_formatter.writer.add_scalar(
                "average_speed_train/average_speed_mean",
                average_speed,
                self.num_timesteps,
            )

            if sum(list_of_x_pos) != 0:
                for i in range(len(list_of_x_pos)):
                    self.tb_formatter.writer.add_scalar(
                        f"positions/x_pos_of_{i + 1}",
                        list_of_x_pos[i],
                        self.num_timesteps,
                    )

            if sum(list_of_x_velocities) != 0:
                for i in range(len(list_of_x_velocities)):
                    self.tb_formatter.writer.add_scalar(
                        f"velocities/x_velocity_of_{i + 1}",
                        list_of_x_velocities[i],
                        self.num_timesteps,
                    )

            for i in range(len(list_of_av_speeds)):
                self.tb_formatter.writer.add_scalar(
                    f"average_speed_train/average_speed_env_{i+1}",
                    list_of_av_speeds[i],
                    self.num_timesteps,
                )

            self.tb_formatter.writer.flush()
            return True
