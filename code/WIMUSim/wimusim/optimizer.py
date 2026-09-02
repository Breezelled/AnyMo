import warnings
import pickle
from tqdm import tqdm
from typing import Dict, Tuple
import wandb
import torch
import pytorch3d.transforms.rotation_conversions as rc
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wimusim.utils import interpolate_quaternions_slerp


class Optimizer:
    def __init__(self, env, meta_info=None):
        self.env = env
        self.loss_coeff_dict = {
            "rmse": 1,
            "rom": 1,
            "sym": 1,
            "b_range": 1,
            "p_range": 1,
            "temp_reg": 1e-2,
            "h_noise_dist": 1,
            "h_std": 1,
            "do_norm": 1,  # ensure the quaternion norm is 1
        }  # This can also be optimization parameter
        self.optimizers_list = []

        self.loss_dict: Dict[str, torch.Tensor] = dict(
            rmse=torch.tensor(0.0, device=env.device),
            rom=torch.tensor(0.0, device=env.device),
            b_range=torch.tensor(0.0, device=env.device),
            p_range=torch.tensor(0.0, device=env.device),
            sym=torch.tensor(0.0, device=env.device),
            temp_reg=torch.tensor(0.0, device=env.device),
            h_noise_dist=torch.tensor(0.0, device=env.device),
            h_std=torch.tensor(0.0, device=env.device),
        )

        self.target_imu_dict = {}
        self.rmse_weight_dict = {imu_name: (1.0, 2.0) for imu_name in env.P.imu_names}
        self.optimizer_dict = {}
        self.scheduler_dict = {}
        self.epoch_log = {}

        self.eps = torch.finfo(torch.float32).eps

        if meta_info is None:
            self.meta_info = {}
        else:
            self.meta_info = meta_info

    def init_optimizers(
        self,
        config: dict = None,
    ):
        if config is None:
            # Default configuration
            config = {
                "B_lr": 1e-3,
                "Pp_lr": 1e-3,
                "Po_lr": 1e-3,
                "Do_lr": 1e-4,
                "Dt_lr": 1e-3,
                "Hb_lr": 1e-3,
                "Heta_lr": 1e-3,
            }

        self.optimizer_dict = {
            "B": torch.optim.Adam(self.env.B.rp.values(), lr=config["B_lr"]),
            "Do": torch.optim.Adam(self.env.D.orientation.values(), lr=config["Do_lr"]),
            "Dt": torch.optim.Adam(self.env.D.translation.values(), lr=config["Dt_lr"]),
            "Pp": torch.optim.Adam(self.env.P.rp.values(), lr=config["Pp_lr"]),
            "Po": torch.optim.Adam(self.env.P.ro.values(), lr=config["Po_lr"]),
            "Hb": torch.optim.Adam(
                list(self.env.H.ba.values()) + list(self.env.H.bg.values()),
                lr=config["Hb_lr"],
            ),
            "Heta": torch.optim.Adam(
                list(self.env.H.eta_a.values()) + list(self.env.H.eta_g.values()),
                lr=config["Heta_lr"],
            ),
        }

        # May not be really necessary. Just keep the patience for early stopping large enough
        self.scheduler_dict = {
            "Do": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_dict["Do"], T_0=2, T_mult=2, eta_min=1e-7, verbose=False
            ),
            "Do": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_dict["Do"], T_0=2, T_mult=4, eta_min=1e-8, verbose=False
            ),
        }

    def set_target_IMU_dict(self, target_imu_dict):
        """
        Set the target IMU data for the optimization.
        """
        # TODO: implement validation

        self.target_imu_dict = target_imu_dict

    def compare_real_sim_IMU_data(self, interval=(1000, 2000)):
        """
        Compare the real and simulated IMU data for the given interval.

        :param interval:
        :return:
        """
        start, end = interval[0], interval[1]
        for imu_name in self.target_imu_dict.keys():
            fig, axs = plt.subplots(3, 1, figsize=(15, 5))
            fig.suptitle(
                f"{imu_name} - Acc Real-Sim Comparison - ID {interval[0]} to {interval[1]}"
            )
            for i in range(3):
                axs[i].plot(
                    self.target_imu_dict[imu_name][0][start:end, i].cpu().numpy(),
                    label="real",
                )
                axs[i].plot(
                    self.env.simulated_IMU_dict[imu_name][0]
                    .detach()
                    .cpu()
                    .numpy()[start:end, i],
                    label="sim",
                )
                axs[i].legend()

            plt.show()

            fig, axs = plt.subplots(3, 1, figsize=(15, 5))
            fig.suptitle(
                f"{imu_name} - Gyro Real-Sim Comparison - ID {interval[0]} to {interval[1]}"
            )
            for i in range(3):
                axs[i].plot(
                    self.target_imu_dict[imu_name][1][start:end, i].cpu().numpy(),
                    label="real",
                )
                axs[i].plot(
                    self.env.simulated_IMU_dict[imu_name][1]
                    .detach()
                    .cpu()
                    .numpy()[start:end, i],
                    label="sim",
                )
                axs[i].legend()
            plt.show()

    def log_IMU_data_viz_comparison(self, segment_length=1000):
        """
        Log the comparison between the real and simulated IMU data.
        wandb must be initialized before calling this function.
        :param segment_length:
        :return:
        """
        # Check if the wandb is initialized
        if wandb.run is None:
            raise ValueError(
                "Wandb is not initialized. Please initialize wandb before calling this function."
            )

        data_length = self.env.D.n_samples

        # Choose the start and end point randomly from the data
        # Ensure the segment length is not greater than the data length
        if segment_length > data_length:
            print("Segment length exceeds data length. Adjusting to full data length.")
            segment_length = data_length

        # Calculate the number of possible complete segments
        num_segments = data_length // segment_length

        # If no complete segments possible, adjust the segment length to full data length
        if num_segments == 0:
            print(
                "Data length less than segment length. Adjusting to full data length."
            )
            start = 0
            end = data_length
        else:
            # Randomly choose one of the possible segment indices
            start_index = np.random.randint(0, num_segments)
            start = start_index * segment_length
            end = start + segment_length

        for imu_name in self.target_imu_dict.keys():
            # Acceleration comparison
            fig, axs = plt.subplots(3, 1, figsize=(12, 5))
            fig.tight_layout()
            fig.suptitle(f"{imu_name} - Acc Real-Sim Comparison - ID {start} to {end}")
            for i in range(3):
                real_data = (
                    self.target_imu_dict[imu_name][0][start:end, i].cpu().numpy()
                )
                sim_data = (
                    self.env.simulated_IMU_dict[imu_name][0]
                    .detach()
                    .cpu()
                    .numpy()[start:end, i]
                )
                axs[i].plot(real_data, label="real")
                axs[i].plot(sim_data, label="sim")
                axs[i].legend()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf)
            wandb.log(
                {f"{imu_name} - Acceleration Comparison": wandb.Image(image)},
                commit=False,
            )
            buf.close()
            plt.close(fig)

            # Gyroscope comparison
            fig, axs = plt.subplots(3, 1, figsize=(15, 5))
            fig.tight_layout()
            fig.suptitle(f"{imu_name} - Gyro Real-Sim Comparison - ID {start} to {end}")
            for i in range(3):
                real_data = (
                    self.target_imu_dict[imu_name][1][start:end, i].cpu().numpy()
                )
                sim_data = (
                    self.env.simulated_IMU_dict[imu_name][1]
                    .detach()
                    .cpu()
                    .numpy()[start:end, i]
                )
                axs[i].plot(real_data, label="real")
                axs[i].plot(sim_data, label="sim")
                axs[i].legend()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image = Image.open(buf)
            wandb.log(
                {f"{imu_name} - Gyroscope Comparison": wandb.Image(image)}, commit=False
            )
            buf.close()
            plt.close(fig)

    def fit(
        self,
        epochs: int = 10000,
        optimizers_list=None,
        loss_coeff_dict=None,
        early_stopping=True,
        patience=20,
        tolerance=1e-4,
        log_wandb=False,
        wandb_project_config=None,
    ):
        """
        Fit the model to the environment with early stopping based on training loss.
        :param epochs: Number of epochs to train.
        :param optimizers_list: List of optimizers to use for training.
        :param patience: Number of epochs to tolerate no improvement.
        :param tolerance: Minimum change in loss to consider as an improvement.
        :param log_wandb: If True, log the training progress to wandb.
        :param wandb_run_name: Name of the wandb run.
        """
        loss_log = []
        best_loss = float("inf")
        epochs_no_improve = 0  # for early stopping

        if log_wandb:
            if wandb_project_config is None:
                wandb_project_name = "test_realworld"
                wandb_run_name = "test_run"
            else:
                wandb_project_name = wandb_project_config["project_name"]
                wandb_run_name = wandb_project_config["run_name"]

            wandb.init(
                project=wandb_project_name,
                name=wandb_run_name,
                config={**self.meta_info, **self.loss_coeff_dict},
            )

        if optimizers_list is None:
            optimizers_list = list(self.optimizer_dict.values())

        for epoch_i in tqdm(range(epochs)):
            self.epoch_log = {}  # Initialize the epoch log dict
            for optim in self.optimizer_dict.values():
                optim.zero_grad()

            loss_i = self.calc_losses(
                loss_coeff_dict=loss_coeff_dict, log_wandb=log_wandb
            )

            loss_i.backward()
            loss_log.append(loss_i.item())

            # Check if there are NaN in D's gradients
            # torch.nn.utils.clip_grad_value_(self.env.D.orientation.values(), 0.001)
            self._check_D_gradients()

            for optim in optimizers_list:
                optim.step()
            for scheduler in self.scheduler_dict.values():
                scheduler.step(epoch_i)

            if log_wandb:
                # Save comparison images every 1000 epochs
                if epoch_i % 1000 == 0:
                    self.log_IMU_data_viz_comparison()
                wandb.log(self.epoch_log)

            if early_stopping:
                # Check for improvement
                if loss_i < best_loss - tolerance:
                    best_loss = loss_i
                    epochs_no_improve = 0

                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping after {epoch_i + 1} epochs.")
                        break
        if log_wandb:
            wandb.finish()
        return loss_log

    def pre_fit_Po(self, epochs=100, lr=1e-3):
        P_ro_before = {k: v.clone().detach() for k, v in self.env.P.ro.items()}

        # Adjust the Pro params (convert it in degrees)
        coeff_dict = self.loss_coeff_dict.copy()
        coeff_dict["p_range"] = 0

        self.optimizer_dict["Po"].param_groups[0]["lr"] = lr
        self.fit(
            epochs=epochs,
            optimizers_list=[self.optimizer_dict["Po"]],
            loss_coeff_dict=coeff_dict,
            patience=10,
            tolerance=1e-1,
        )

        # Check the new P
        print("IMU Name\t\tBefore Adjustment\t\t=>\t\tAfter Adjustment")
        for key, p_ro_after in self.env.P.ro.items():
            before_deg = torch.rad2deg(P_ro_before[key]).detach().cpu().numpy()
            after_deg = torch.rad2deg(p_ro_after).detach().cpu().numpy()
            before_str = ", ".join(f"{val:7.2f}" for val in before_deg)
            after_str = ", ".join(f"{val:7.2f}" for val in after_deg)
            print(f"{key[1]:10}:\t{before_str}\t=>\t{after_str}")
        pass

    def _check_D_gradients(self):
        # Check if any gradients are NaN after the specified loss computation
        window_size = 30 * 5  # 5 seconds before and after the NaN value
        for joint_name, q_joint in self.env.D.orientation.items():
            if q_joint.grad is not None and torch.isnan(q_joint.grad).any():
                warnings.warn(
                    f"NaN values detected in {joint_name}.grad. D.orientation have been modified."
                )
                # Save the optimized parameters
                random_int = np.random.randint(1000)
                pkl_path = f"./opt-debug-{random_int}.pkl"
                with open(pkl_path, "wb") as file:
                    pickle.dump(self, file)
                print("Optimized parameters for debugging are saved to", pkl_path)
                # Check if there are NaN values in the gradients
                # If exists, delete the values in the window around the NaN values
                # then interpolate with the adjuscent values
                mask = torch.isnan(q_joint.grad).any(dim=-1)
                for idx in torch.nonzero(mask).squeeze(-1):
                    start_idx = max(0, idx - window_size)
                    end_idx = min(mask.size(0), idx + window_size + 1)
                    mask[start_idx:end_idx] = True

                q_joint_interpolated = interpolate_quaternions_slerp(q_joint, mask)
                self.env.D.orientation[joint_name] = (
                    q_joint_interpolated.detach()
                    .clone()
                    .to(self.env.device)
                    .requires_grad_(True)
                )
                # Calculate the loss again
                for optim in self.optimizer_dict.values():
                    optim.zero_grad()
                self.calc_losses()

                # q_joint.grad = torch.where(
                #     torch.isnan(q_joint.grad),
                #     torch.zeros_like(q_joint.grad),
                #     q_joint.grad,
                # )
                # mask = torch.isnan(q_joint.grad).any()

    def calc_losses(
        self, loss_coeff_dict: Dict[str, float] = None, log_wandb=False
    ) -> torch.Tensor:
        if loss_coeff_dict is None:
            loss_coeff_dict = self.loss_coeff_dict

        # To check what's happening here. (sometimes the gradients becomes NaN)
        # Keep the following until the issue is fully resolved
        # for optim in self.optimizer_dict.values():
        #     optim.zero_grad()
        # rom_loss_info = self.calc_rom_loss()
        # self.loss_dict["rom"].backward()
        # self._check_D_gradients()
        #
        # for optim in self.optimizer_dict.values():
        #     optim.zero_grad()

        # Run env.simulate() everytime when calculating the rmse loss
        rmse_loss = self.calc_rmse_loss(self.target_imu_dict, run_simulate=True)
        sym_loss_info = self.calc_symmetry_loss()
        temp_reg_loss_info = self.calc_temp_reg_loss()
        b_range_loss_info = self.calc_B_range_loss()
        p_range_loss_info = self.calc_P_range_loss()
        rom_loss_info = self.calc_rom_loss()
        h_noise_dict_loss_info = self.calc_H_noise_dist_loss()
        h_std_loss_info = self.calc_H_std_loss()
        do_norm_loss_info = self.calc_Do_norm_loss()  # Just log its total

        assert (
            loss_coeff_dict.keys() == self.loss_dict.keys()
        ), "Inconsistent loss names."

        loss_total = torch.tensor(0.0, device=self.env.device)
        for loss_name, loss_value in self.loss_dict.items():
            loss_total += loss_value * loss_coeff_dict[loss_name]

        lr_dict = {
            f"lr_{opt_name}": opt_obj.param_groups[0]["lr"]
            for opt_name, opt_obj in self.optimizer_dict.items()
        }

        if log_wandb:
            # Log the losses to wandb
            self.epoch_log = {
                **self.epoch_log,  # ROM Loss
                "loss_total": loss_total.detach().cpu().numpy(),
                **{
                    loss_name: loss_val.detach().cpu().numpy()
                    for loss_name, loss_val in self.loss_dict.items()
                },
                **lr_dict,
                **{
                    f"RMSE_{imu_name}_acc": rmse[0].detach().cpu().numpy()
                    for imu_name, rmse in rmse_loss.items()  # imu_name: (rmse_acc, rmse_gyro)
                },
                **{
                    f"RMSE_{imu_name}_gyro": rmse[1].detach().cpu().numpy()
                    for imu_name, rmse in rmse_loss.items()
                },
                **{
                    f"ROM_{joint}_{cond}_{axis}": val
                    for joint in rom_loss_info.keys()
                    for cond, direction in [("lt_min", "XYZ"), ("gt_max", "XYZ")]
                    for axis, val in zip(
                        direction, rom_loss_info[joint][cond].detach().cpu().numpy()
                    )
                },
                # ROM cond-axis combined
                **{
                    f"ROM_{joint}": sum(rom_loss_info[joint].values())
                    .sum()
                    .detach()
                    .cpu()
                    .numpy()
                    for joint in rom_loss_info.keys()
                },
                # B Range Loss
                **{
                    f"B_{'2'.join(edge)}_{cond}_{axis}": val
                    for edge in b_range_loss_info.keys()
                    for cond, direction in [("lt_min", "XYZ"), ("gt_max", "XYZ")]
                    for axis, val in zip(
                        direction, b_range_loss_info[edge][cond].detach().cpu().numpy()
                    )
                },
                # B cond-axis combined
                **{
                    f"B_{'2'.join(edge)}": sum(b_range_loss_info[edge].values())
                    .sum()
                    .detach()
                    .cpu()
                    .numpy()
                    for edge in b_range_loss_info.keys()
                },
                ## P_rp Range Loss
                **{
                    f"P_rp_{'2'.join(edge)}_{cond}_{axis}": val
                    for edge in p_range_loss_info[0].keys()
                    for cond, direction in [("lt_min", "XYZ"), ("gt_max", "XYZ")]
                    for axis, val in zip(
                        direction,
                        p_range_loss_info[0][edge][cond].detach().cpu().numpy(),
                    )
                },
                **{
                    f"P_rp_{'2'.join(edge)}": sum(p_range_loss_info[0][edge].values())
                    .sum()
                    .detach()
                    .cpu()
                    .numpy()
                    for edge in p_range_loss_info[0].keys()
                },
                ## P_ro Range Loss
                **{
                    f"P_ro_{'2'.join(edge)}_{cond}_{axis}": val
                    for edge in p_range_loss_info[0].keys()
                    for cond, direction in [("lt_min", "XYZ"), ("gt_max", "XYZ")]
                    for axis, val in zip(
                        direction,
                        p_range_loss_info[1][edge][cond].detach().cpu().numpy(),
                    )
                },
                **{
                    f"P_ro_{'2'.join(edge)}": sum(p_range_loss_info[1][edge].values())
                    .sum()
                    .detach()
                    .cpu()
                    .numpy()
                    for edge in p_range_loss_info[1].keys()
                },
                # Symmetry loss
                **{
                    f"Sym_{'2'.join(pair)}_{axis}": loss_axis.detach().cpu().numpy()
                    for pair, loss in sym_loss_info.items()
                    for loss_axis, axis in zip(loss, "XYZ")
                },
                # Temporal regularization loss
                **{
                    f"temp_reg_{joint}": loss.detach().cpu().numpy()
                    for joint, loss in temp_reg_loss_info[0].items()
                },
                **{
                    f"temp_reg_{joint}": loss.detach().cpu().numpy()
                    for joint, loss in temp_reg_loss_info[1].items()
                },
                # H_std_range_loss
                **{
                    f"H_{imu_name}_acc_std": loss.detach().cpu().numpy()
                    for imu_name, loss in h_std_loss_info[0].items()
                },
                **{
                    f"H_{imu_name}_gyro_std": loss.detach().cpu().numpy()
                    for imu_name, loss in h_std_loss_info[1].items()
                },
                **{
                    f"H_{imu_name}_acc_noise_mean": loss["mean_loss"]
                    .detach()
                    .cpu()
                    .numpy()
                    for imu_name, loss in h_noise_dict_loss_info[0].items()
                },
                **{
                    f"H_{imu_name}_acc_noise_std": loss["std_loss"]
                    .detach()
                    .cpu()
                    .numpy()
                    for imu_name, loss in h_noise_dict_loss_info[0].items()
                },
                **{
                    f"H_{imu_name}_acc_noise_freq": loss["freq_loss"]
                    .detach()
                    .cpu()
                    .numpy()
                    for imu_name, loss in h_noise_dict_loss_info[0].items()
                },
                **{
                    f"H_{imu_name}_gyro_noise_mean": loss["mean_loss"]
                    .detach()
                    .cpu()
                    .numpy()
                    for imu_name, loss in h_noise_dict_loss_info[1].items()
                },
                **{
                    f"H_{imu_name}_gyro_noise_mean": loss["std_loss"]
                    .detach()
                    .cpu()
                    .numpy()
                    for imu_name, loss in h_noise_dict_loss_info[1].items()
                },
                **{
                    f"H_{imu_name}_gyro_noise_freq": loss["freq_loss"]
                    .detach()
                    .cpu()
                    .numpy()
                    for imu_name, loss in h_noise_dict_loss_info[1].items()
                },
                **{  # sometime P_ro fall in singularity and the P.ro becomes NaN
                    f"P_{'2'.join(edge)}_{axis}": np.rad2deg(rad)
                    for edge in self.env.P.ro.keys()
                    for rad, axis in zip(
                        self.env.P.ro[edge].detach().cpu().numpy(), "XYZ"
                    )
                },
            }

        return loss_total

    def calc_rmse_loss(
        self,
        target_imu_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None,
        run_simulate: bool = False,
    ):
        """
        Calculate the RMSE loss between target and simulated IMU data.
        All the dicts should have the same keys.

        :param target_imu_dict: dict of target IMU data {imu_name: (acc, gyro)}
        :param run_simulate: If True, simulate the environment before calculating the loss.
        :return:
        """

        if target_imu_dict is None:
            if self.target_imu_dict is None:
                raise ValueError("Target IMU data is not available.")
            target_imu_dict = self.target_imu_dict

        criterion = torch.nn.MSELoss()
        if self.env.simulated_IMU_dict is None:
            warnings.warn("Simulated IMU data is not available. Simulating now...")
            self.env.simulate()

        # Set the simulated IMU dict
        if run_simulate:
            simulated_imu_dict = self.env.simulate()
        else:
            simulated_imu_dict = self.env.simulated_IMU_dict

        assert target_imu_dict.keys() == simulated_imu_dict.keys()
        assert simulated_imu_dict.keys() == self.rmse_weight_dict.keys()

        # Calculate RMSE for each IMU
        rmse_loss_dict = {}
        for imu_name in target_imu_dict.keys():
            acc_rmse = torch.sqrt(
                criterion(
                    target_imu_dict[imu_name][0],
                    simulated_imu_dict[imu_name][0],
                )
            )
            gyro_rmse = torch.sqrt(
                criterion(
                    target_imu_dict[imu_name][1],
                    simulated_imu_dict[imu_name][1],
                )
            )
            rmse_loss_dict[imu_name] = (acc_rmse, gyro_rmse)

        # Calculate the total RMSE loss using the weights
        rmse_loss_total = 0.0
        for imu_name, (acc_rmse, gyro_rmse) in rmse_loss_dict.items():
            rmse_loss_total += (
                acc_rmse * self.rmse_weight_dict[imu_name][0]
                + gyro_rmse * self.rmse_weight_dict[imu_name][1]
            )

        self.loss_dict["rmse"] = rmse_loss_total

        return rmse_loss_dict

    def calc_Do_norm_loss(self):
        """

        :return:
        """
        loss_Do_norm_dict = {}
        for joint_name, q_joint in self.env.D.orientation.items():
            # Calculate norm of the quaternions for this joint
            norm_q_joint = torch.norm(
                q_joint, p=2, dim=1
            )  # Assuming q_joint is of shape [N, 4]

            # Compute the loss for this joint as the squared difference from 1
            loss_Do_norm_dict[joint_name] = torch.mean((1.0 - norm_q_joint) ** 2)

        # Sum up the losses from all joints
        do_norm_total = sum(loss_Do_norm_dict.values())
        self.loss_dict["do_norm"] = do_norm_total

        return loss_Do_norm_dict

    def calc_rom_loss(self):
        """
        Calculate the ROM loss between the joint angles and the ROM limits.

        :return:
        """
        assert self.env.D.orientation.keys() == self.env.B.rom_dict.keys()

        rom_loss_dict = {}
        order_dict = {"X": 0, "Y": 1, "Z": 2}
        for joint_name, q_joint in self.env.D.orientation.items():
            # might need to use different convention for different joints
            convention = "XYZ"
            axis_order = [order_dict[conv] for conv in convention]

            if torch.isnan(q_joint).any():
                print(joint_name)
                raise ValueError("NaN values in joint angles.")

            joint_angles = rc.matrix_to_euler_angles(
                rc.quaternion_to_matrix(q_joint), convention
            )
            if torch.isnan(joint_angles).any():
                print(joint_name)
                raise ValueError("NaN values in joint angles.")
            rom_min = self.env.B.rom_dict[joint_name][:, 0][axis_order]
            rom_max = self.env.B.rom_dict[joint_name][:, 1][axis_order]

            # Check for NaNs
            if torch.isnan(rom_min).any() or torch.isnan(rom_max).any():
                print(f"NaNs found in ROM limits for {joint_name}")
                continue

            lt_min_mask = joint_angles < rom_min
            lt_min_diff = joint_angles - rom_min

            gt_max_mask = joint_angles > rom_max
            gt_max_diff = joint_angles - rom_max

            lt_min = (
                torch.where(lt_min_mask, lt_min_diff, 0.0).abs().sum(dim=0)
                / joint_angles.shape[0]
            )
            gt_max = (
                torch.where(gt_max_mask, gt_max_diff, 0.0).abs().sum(dim=0)
                / joint_angles.shape[0]
            )

            # Check for NaNs in computed values
            if torch.isnan(lt_min).any() or torch.isnan(gt_max).any():
                print(f"NaNs found in lt_min or gt_max for {joint_name}")
                continue

            rom_loss_dict[joint_name] = {"lt_min": lt_min, "gt_max": gt_max}

        rom_loss_total = torch.tensor(0.0, device=self.env.device)
        for joint_loss in rom_loss_dict.values():
            rom_loss_total = (
                rom_loss_total + joint_loss["lt_min"].sum() + joint_loss["gt_max"].sum()
            )

        self.loss_dict["rom"] = rom_loss_total

        return rom_loss_dict

    def calc_B_range_loss(self):
        """
        Calculate the range loss for the body parameters.

        :return:
        """
        b_range_loss_dict = {}
        for pos_name, rp_value in self.env.B.rp.items():
            rp_min = self.env.B.rp_range_dict[pos_name][:, 0]  # (3, )
            rp_max = self.env.B.rp_range_dict[pos_name][:, 1]  # (3, )
            assert all(
                rp_min < rp_max
            ), f"Invalid range for {pos_name}: {rp_min} - {rp_max}"
            lt_min_mask = rp_value < rp_min
            lt_min_diff = rp_value - rp_min

            gt_max_mask = rp_value > rp_max
            gt_max_diff = rp_value - rp_max

            lt_min = torch.where(lt_min_mask, lt_min_diff, 0.0).abs()
            gt_max = torch.where(gt_max_mask, gt_max_diff, 0.0).abs()
            b_range_loss_dict[pos_name] = {"lt_min": lt_min, "gt_max": gt_max}

        # Calculate the total b_range loss
        b_range_loss_total = torch.tensor(0.0, device=self.env.device)
        for b_range_loss in b_range_loss_dict.values():
            b_range_loss_total += b_range_loss["lt_min"].sum()
            b_range_loss_total += b_range_loss["gt_max"].sum()
        self.loss_dict["b_range"] = b_range_loss_total

        return b_range_loss_dict

    def calc_P_range_loss(self):
        """
        Calculate the range loss for the placement parameters.
        :return:
        """
        # Calculate the range loss for the relative positions
        p_rp_range_loss_dict = {}
        for pos_name, rp_value in self.env.P.rp.items():
            rp_min: torch.Tensor = self.env.P.rp_range_dict[pos_name][:, 0]  # (3, )
            rp_max: torch.Tensor = self.env.P.rp_range_dict[pos_name][:, 1]  # (3, )
            assert all(rp_min < rp_max)
            lt_min_mask = rp_value < rp_min
            lt_min_diff = rp_value - rp_min

            gt_max_mask = rp_value > rp_max
            gt_max_diff = rp_value - rp_max
            lt_min = torch.where(lt_min_mask, lt_min_diff, 0.0).abs()
            gt_max = torch.where(gt_max_mask, gt_max_diff, 0.0).abs()
            p_rp_range_loss_dict[pos_name] = {"lt_min": lt_min, "gt_max": gt_max}

        # Calculate the range loss for the relative orientations
        p_ro_range_loss_dict = {}
        for pos_name, ro_value in self.env.P.ro.items():
            ro_min: torch.Tensor = self.env.P.ro_range_dict[pos_name][:, 0]
            ro_max: torch.Tensor = self.env.P.ro_range_dict[pos_name][:, 1]
            assert all(ro_min < ro_max)
            lt_min_mask = rp_value < rp_min
            lt_min_diff = rp_value - rp_min

            gt_max_mask = rp_value > rp_max
            gt_max_diff = rp_value - rp_max

            lt_min = torch.where(lt_min_mask, lt_min_diff, 0.0).abs()
            gt_max = torch.where(gt_max_mask, gt_max_diff, 0.0).abs()
            p_ro_range_loss_dict[pos_name] = {"lt_min": lt_min, "gt_max": gt_max}

        p_range_loss_total = torch.tensor(0.0, device=self.env.device)
        for rp_loss_dict in p_rp_range_loss_dict.values():
            p_range_loss_total += rp_loss_dict["lt_min"].sum()
            p_range_loss_total += rp_loss_dict["gt_max"].sum()
        for ro_loss_dict in p_ro_range_loss_dict.values():
            p_range_loss_total += ro_loss_dict["lt_min"].sum()
            p_range_loss_total += ro_loss_dict["gt_max"].sum()
        self.loss_dict["p_range"] = p_range_loss_total

        return p_rp_range_loss_dict, p_ro_range_loss_dict

    def calc_symmetry_loss(self):
        """
        Calculate the symmetry loss.
        :return:
        """

        if self.env.B.symmetry_key_pairs is None:
            self.loss_dict["sym"] = torch.tensor(0.0, device=self.env.device)
            return {}

        symmetry_loss_dict = {}
        for key_R, key_L in self.env.B.symmetry_key_pairs:
            left_mirrored = self.env.B.rp[key_L].clone()
            left_mirrored[0] = -self.env.B.rp[key_L][0]  # flip around x-axis
            key_from, key_to = key_R

            symmetry_loss_dict[
                (key_from.replace("R_", ""), key_to.replace("R_", ""))
            ] = torch.abs(self.env.B.rp[key_R] - left_mirrored)

        symmetry_loss_total = torch.tensor(0.0, device=self.env.device)
        for symmetry_loss in symmetry_loss_dict.values():
            symmetry_loss_total += symmetry_loss.sum()
        self.loss_dict["sym"] = symmetry_loss_total

        return symmetry_loss_dict

    def calc_temp_reg_loss(self, nth_diff: int = 3):
        """
        Calculate the temporal regularization loss for D parameter.
        It converts the orientation to euler angles and calculates the temporal regularization loss.

        :param nth_diff:
        :return:
        """
        temp_reg_loss_ori_dict = {}
        # Temporal regularization loss from the changes in orientations
        # Might be better to use the quaternion difference directly
        for joint_name in self.env.D.orientation.keys():
            temp_reg_loss_ori_dict[joint_name] = torch.mean(
                torch.square(
                    torch.diff(self.env.D.orientation[joint_name], dim=0, n=nth_diff)
                )
            )

        temp_reg_loss_trans_dict = {}
        # Temporal regularization loss from the changes in translation
        for joint_name in self.env.D.translation.keys():
            temp_reg_loss_trans_dict[joint_name] = torch.mean(
                torch.square(
                    torch.diff(self.env.D.translation[joint_name], dim=0, n=nth_diff)
                )
            )

        temp_reg_loss_total = torch.tensor(0.0, device=self.env.device)
        for reg_loss_ori in temp_reg_loss_ori_dict.values():
            temp_reg_loss_total += reg_loss_ori
        for reg_loss_trans in temp_reg_loss_trans_dict.values():
            temp_reg_loss_total += reg_loss_trans

        self.loss_dict["temp_reg"] = temp_reg_loss_total

        return temp_reg_loss_ori_dict, temp_reg_loss_trans_dict

    @staticmethod
    def calc_white_noise_loss(ts_data):
        """
        Calculate the white noise loss of the time series data.
        :param ts_data: (T, N) tensor
        :return:
        """
        frequency_data = torch.fft.fft(ts_data, dim=0)
        frequency_energy = torch.abs(frequency_data) ** 2
        mean_energy = torch.mean(frequency_energy)
        std_energy = torch.std(frequency_energy)
        return std_energy / (mean_energy + torch.finfo(torch.float32).eps)

    def calc_H_noise_dist_loss(self):
        """
        Calculate the noise distribution loss for the accelerometer and gyroscope.
        """

        H_acc_noise_loss_dict = {}
        H_gyro_noise_loss_dict = {}
        for imu_placement in self.env.H.eta_a.keys():
            H_acc_noise_loss_dict[imu_placement] = {}

            H_acc_noise_loss_dict[imu_placement]["mean_loss"] = torch.mean(
                torch.mean(self.env.H.eta_a[imu_placement], dim=0) ** 2
            )
            H_acc_noise_loss_dict[imu_placement]["std_loss"] = torch.mean(
                (
                    self.env.H.sa[imu_placement]
                    - torch.std(self.env.H.eta_a[imu_placement], dim=0)
                )
                ** 2
            )

        for imu_placement in self.env.H.eta_g.keys():
            H_gyro_noise_loss_dict[imu_placement] = {}
            H_gyro_noise_loss_dict[imu_placement]["mean_loss"] = torch.mean(
                torch.mean(self.env.H.eta_g[imu_placement], dim=0) ** 2
            )
            H_gyro_noise_loss_dict[imu_placement]["std_loss"] = torch.mean(
                (
                    self.env.H.sg[imu_placement]
                    - torch.std(self.env.H.eta_g[imu_placement], dim=0)
                )
                ** 2
            )

        for imu_placement in self.env.H.eta_a.keys():
            freq_loss_acc = self.calc_white_noise_loss(self.env.H.eta_a[imu_placement])
            H_acc_noise_loss_dict[imu_placement]["freq_loss"] = freq_loss_acc

        for imu_placement in self.env.H.eta_g.keys():
            freq_loss_gyro = self.calc_white_noise_loss(self.env.H.eta_g[imu_placement])
            H_gyro_noise_loss_dict[imu_placement]["freq_loss"] = freq_loss_gyro

        noise_dist_loss_total = torch.tensor(0.0, device=self.env.device)
        for acc_noise_loss in H_acc_noise_loss_dict.values():
            noise_dist_loss_total += (
                acc_noise_loss["mean_loss"]
                + acc_noise_loss["std_loss"]
                + acc_noise_loss["freq_loss"]
            )
        for gyro_noise_loss in H_gyro_noise_loss_dict.values():
            noise_dist_loss_total += (
                gyro_noise_loss["mean_loss"]
                + gyro_noise_loss["std_loss"]
                + gyro_noise_loss["freq_loss"]
            )

        self.loss_dict["h_noise_dist"] = noise_dist_loss_total

        return H_acc_noise_loss_dict, H_gyro_noise_loss_dict

    def calc_H_std_loss(self):
        """
        Calculate the loss for the standard deviation of the noise.
        ** Currently, simply penalize bigger std values.
        """

        H_acc_std_loss_dict = {}
        H_gyro_std_loss_dict = {}

        for imu_placement, H_sa in self.env.H.sa.items():
            H_acc_std_loss_dict[imu_placement] = H_sa**2

        for imu_placement, H_sg in self.env.H.sg.items():
            H_gyro_std_loss_dict[imu_placement] = H_sg**2

        h_std_loss_total = torch.tensor(0.0, device=self.env.device)
        for sa_range_loss in H_acc_std_loss_dict.values():
            h_std_loss_total += sa_range_loss.sum()
        for sg_range_loss in H_gyro_std_loss_dict.values():
            h_std_loss_total += sg_range_loss.sum()
        self.loss_dict["h_std"] = h_std_loss_total

        return H_acc_std_loss_dict, H_gyro_std_loss_dict
