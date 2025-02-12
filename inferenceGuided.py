import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted

from canopyPlots import createCHM
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from forwardImaging import ForwardImaging


@dataclass
class Configuration:
    resolution: int
    use_side_info: bool
    load_cube: bool


def parse_arguments() -> Configuration:
    """Parse command-line arguments and return a Configuration object."""
    parser = argparse.ArgumentParser(
        description="Replicate diffusion results for canopy height model."
    )
    parser.add_argument(
        "--resolution", type=int, default=3, help="Generated image resolution."
    )
    parser.add_argument(
        "--use_side_info",
        type=int,
        default=0,
        help="Flag to use side information (1 to use, 0 otherwise).",
    )
    parser.add_argument(
        "--load_cube",
        type=int,
        default=0,
        help="Flag to load cube (1 to load, 0 otherwise).",
    )
    args = parser.parse_args()
    return Configuration(
        resolution=args.resolution,
        use_side_info=(args.use_side_info == 1),
        load_cube=(args.load_cube == 1),
    )


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D cube by dividing by the maximum value (per spatial position).

    Args:
        cube (np.ndarray): Input 3D data array.

    Returns:
        np.ndarray: Normalized data.
    """
    max_vals = cube.max(axis=0)
    return cube / (max_vals + 1e-12)


def load_data(config: Configuration) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the low-resolution input and high-resolution ground truth cubes from the testCubes folder.
    
    The files are expected to be named as follows:
        - Low-resolution input:  "testCubes/input{resolution}.npy"
        - High-resolution ground truth: "testCubes/gt{resolution}.npy"
    
    Args:
        config (Configuration): Experiment configuration containing the resolution.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The input data and ground truth arrays.
    """
    input_path = f"testCubes/input{config.resolution}.npy"
    gt_path = f"testCubes/gt{config.resolution}.npy"

    input_data = np.load(input_path)
    gt = np.load(gt_path)

    # Rearrange axes to match the desired format.
    input_data = np.swapaxes(np.swapaxes(input_data, -1, -3), -1, -2)
    gt = np.swapaxes(gt, -1, -3)

    print(f"Input data shape: {input_data.shape}, Ground truth shape: {gt.shape}")

    return input_data, gt


def prepare_data(
    input_data: np.ndarray, gt: np.ndarray, config: Configuration
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Normalize, crop, and reorient the data. Convert input data to a torch.Tensor.

    Args:
        input_data (np.ndarray): Low-resolution input data.
        gt (np.ndarray): High-resolution ground truth.
        config (Configuration): Experiment configuration.

    Returns:
        Tuple[torch.Tensor, np.ndarray, np.ndarray]: 
            (input_tensor, gt (normalized and cropped), input_data (for visualization))
    """
    crop_factor = 72  # As used in the original code.
    gt = normalize_cube(gt)[:, : (8 * crop_factor), : (8 * crop_factor)]
    # Note: Input data cropping uses different scale factors.
    input_data = normalize_cube(input_data)[:, : (8 * crop_factor) // 6, : (8 * crop_factor) // 3]

    # Adjust axes.
    input_data = np.swapaxes(input_data, 1, 2)
    gt = np.swapaxes(gt, 1, 2)

    # Convert input data to torch.Tensor and move to GPU.
    input_tensor = torch.tensor(input_data).float().cuda()

    return input_tensor, gt, input_data


def setup_diffusion(config: Configuration) -> Trainer:
    """
    Initialize the UNet model, Gaussian Diffusion process, and Trainer.

    Args:
        config (Configuration): Experiment configuration.

    Returns:
        Trainer: Configured trainer with the diffusion model.
    """
    model = Unet(
        dim=128,
        dim_mults=(8, 16, 16, 16),
        flash_attn=True,
        channels=128,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
        sampling_timesteps=250,  # Using DDIM for faster inference.
    )

    trainer = Trainer(
        diffusion,
        train_batch_size=64,
        train_lr=8e-5,
        train_num_steps=700000,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=True,
        resolution=config.resolution,
    )

    trainer.load(0)
    trainer.ema.ema_model.eval()

    return trainer


def run_guided_diffusion(
    trainer: Trainer, forward_imaging: ForwardImaging, input_tensor: torch.Tensor, gt: np.ndarray
) -> np.ndarray:
    """
    Run the guided diffusion process to generate the diffused output.

    Args:
        trainer (Trainer): The diffusion trainer.
        forward_imaging (ForwardImaging): Forward imaging operator.
        input_tensor (torch.Tensor): Input tensor guiding the diffusion.
        gt (np.ndarray): Ground truth array (used for shape and normalization).

    Returns:
        np.ndarray: The final diffused output in the range [0, 1].
    """
    # Initialize the sample with random noise matching the ground truth shape.
    sample = torch.randn_like(torch.from_numpy(gt).unsqueeze(0).float()).cuda()
    _, channels, _, _ = sample.shape

    # Prepare guidance data.
    y = input_tensor.transpose(-1, -3)
    mask = (y.sum(-1) == 0).float().transpose(-1, -2).unsqueeze(0).repeat(channels, 1, 1).cuda()
    use_mask = mask.sum() > 0

    loss_history = []
    num_steps = 1000

    # Constants for guidance and step size.
    GUIDANCE_DIVISOR = 1.2
    STEP_MULTIPLIER = 20

    pbar = tqdm(
        reversed(range(num_steps)),
        total=num_steps,
        desc="Guided Diffusion"
    )
    for t in pbar:
        # Enable gradient computation for the current sample.
        sample.requires_grad_()

        # Obtain the previous sample and the current prediction (x_start) from the EMA model.
        sample_prev, x_start = trainer.ema.ema_model.p_sample(sample, t)

        # Normalize x_start from the range [-1, 1] to [0, 1].
        x_start_normalized = (x_start + 1) * 0.5

        # Compute the forward imaging transformation.
        distribution, aux = forward_imaging.forward_imaging(x_start_normalized)

        # Compute the base loss from the negative log probability.
        base_loss = -distribution.log_prob(y / GUIDANCE_DIVISOR).mean()

        # If a mask is used, add the penalty term from the auxiliary output.
        mask_penalty = aux[mask == 1].mean() if use_mask else 0.0
        loss = base_loss + mask_penalty

        # Determine the step size adaptively based on the loss.
        loss_value = loss.item()
        step_size = STEP_MULTIPLIER * loss_value
        loss_history.append(loss_value)
        pbar.set_postfix({"loss": loss_value})

        # Compute the gradient of the loss with respect to the current sample.
        grad = torch.autograd.grad(loss, sample)[0]

        # Update the sample by stepping in the direction opposite to the gradient.
        sample = sample_prev - step_size * grad

        # Detach the updated sample from the current computation graph.
        sample = sample.detach()

    # Convert the final sample to numpy and scale to [0, 1].
    diffused = (sample[0].cpu().transpose(-1, -2).detach().numpy() + 1) * 0.5
    return diffused


def plot_results(
    input_data: np.ndarray, gt: np.ndarray, diffused: np.ndarray, config: Configuration, cube_number: int
) -> None:
    """
    Plot the CHM, DTM, zoomed CHM, and profile for input, diffused output, and ground truth.

    Args:
        input_data (np.ndarray): Visualization-ready input data.
        gt (np.ndarray): Ground truth data.
        diffused (np.ndarray): Diffused output.
        config (Configuration): Experiment configuration.
        cube_number (int): Cube index (used for naming the result file).
    """
    # Apply threshold to the diffused output.
    diffused[diffused < 0.02] = 0

    # Crop arrays for visualization.
    crop_width = (8 * 66) // 6
    crop_height = (8 * 66) // 3
    input_vis = input_data[:, :crop_width, :crop_height]
    gt_vis = gt[:, : (8 * 66) // config.resolution, : (8 * 66) // config.resolution]
    diffused_vis = diffused[:, : (8 * 66) // config.resolution, : (8 * 66) // config.resolution]

    # Visualization parameters.
    upper_limit = -1  # Upper limit (negative means full range)
    lower_limit = 0
    zoomx = 16 * 6
    zoomy = 32 * 3
    profile = 100

    chm_gt = createCHM(gt_vis)[0]
    dtm_colormap = "copper"

    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    fig.subplots_adjust(hspace=0, wspace=0)

    # Plot Input.
    chm_input, dtm_input, hs_input = createCHM(input_vis)
    axs[0, 0].imshow(chm_input, interpolation="nearest", aspect=2)
    axs[0, 1].imshow(dtm_input, interpolation="nearest", aspect=2, cmap=dtm_colormap)
    axs[0, 1].imshow(hs_input, interpolation="nearest", aspect=2, cmap="Greys", alpha=0.35)
    axs[0, 2].imshow(
        chm_input[zoomx // 6 : zoomx // 6 + 16, zoomy // 3 : zoomy // 3 + 32],
        interpolation="nearest",
        aspect=2,
    )
    axs[0, 3].imshow(
        np.flip(input_vis[lower_limit // 2 : upper_limit // 2, profile // 6, -200 // 3:], axis=0),
        cmap="gray_r",
        interpolation="nearest",
        aspect=0.25,
    )

    # Plot Diffused.
    chm_diff, dtm_diff, hs_diff = createCHM(diffused_vis)
    axs[1, 0].imshow(chm_diff, interpolation="nearest", vmin=chm_gt.min(), vmax=chm_gt.max())
    axs[1, 1].imshow(dtm_diff, interpolation="nearest", cmap=dtm_colormap)
    axs[1, 1].imshow(hs_diff, interpolation="nearest", cmap="Greys", alpha=0.35)
    axs[1, 2].imshow(
        chm_diff[
            zoomx // config.resolution : zoomx // config.resolution + (96 // config.resolution),
            zoomy // config.resolution : zoomy // config.resolution + (96 // config.resolution),
        ],
        interpolation="nearest",
    )
    axs[1, 3].imshow(
        np.flip(
            diffused_vis[lower_limit:upper_limit, profile // config.resolution, -200 // config.resolution:],
            axis=0,
        ),
        cmap="gray_r",
        interpolation="nearest",
        aspect=1 / config.resolution,
    )

    # Plot Ground Truth.
    chm_gt_, dtm_gt, hs_gt = createCHM(gt_vis)
    axs[2, 0].imshow(chm_gt_, interpolation="nearest")
    axs[2, 1].imshow(dtm_gt, interpolation="nearest", cmap=dtm_colormap)
    axs[2, 1].imshow(hs_gt, interpolation="nearest", cmap="Greys", alpha=0.35)
    axs[2, 2].imshow(
        chm_gt_[
            zoomx // config.resolution : zoomx // config.resolution + (96 // config.resolution),
            zoomy // config.resolution : zoomy // config.resolution + (96 // config.resolution),
        ],
        interpolation="nearest",
    )
    axs[2, 3].imshow(
        np.flip(
            gt_vis[lower_limit:upper_limit, profile // config.resolution, -200 // config.resolution:],
            axis=0,
        ),
        cmap="gray_r",
        interpolation="nearest",
        aspect=1 / config.resolution,
    )

    # Titles and labels.
    axs[0, 0].set_title("CHM")
    axs[0, 1].set_title("DTM")
    axs[0, 2].set_title("CHM Zoom")
    axs[0, 3].set_title("Profile")
    axs[0, 0].set_ylabel("Input")
    axs[1, 0].set_ylabel("Diffusion (Side Info)" if config.use_side_info else "Diffusion")
    axs[2, 0].set_ylabel(f"Ground Truth ({config.resolution}m)")

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    result_filename = f"./results/Diffusion_{config.resolution}_use_side_{config.use_side_info}_{cube_number}_guided.png"
    plt.savefig(result_filename)
    plt.close()
    print(f"Results figure saved to {result_filename}")


def main():
    # Parse configuration.
    config = parse_arguments()
    print(
        f"Configuration: resolution={config.resolution}, "
        f"use_side_info={config.use_side_info}, load_cube={config.load_cube}"
    )

    # Cube index to process.
    cube_number = 37

    # Load the input and ground truth data.
    input_data, gt = load_data(config, cube_number)

    # Prepare data (normalization, cropping, reorientation).
    input_tensor, gt, input_data_prepared = prepare_data(input_data, gt, config)

    # Initialize the diffusion model/trainer.
    trainer = setup_diffusion(config)

    # Instantiate forward imaging.
    forward_imaging = ForwardImaging(config.resolution, trainer.device)

    # Run guided diffusion to obtain the diffused output.
    diffused = run_guided_diffusion(trainer, forward_imaging, input_tensor, gt)

    # Save diffused result.
    diffused_filename = f"diffusion{cube_number}_{config.resolution}_guided.npy"
    np.save(diffused_filename, diffused)
    print(f"Diffused output saved as {diffused_filename}")

    # Plot and save the comparative results.
    plot_results(input_data_prepared, gt, diffused, config, cube_number)


if __name__ == "__main__":
    main()
