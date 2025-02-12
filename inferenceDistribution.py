import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from canopyPlots import createCHM
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from forwardImaging import ForwardImaging

@dataclass
class Configuration:
    resolution: int
    train_batch_size: int
    train_num_steps: int
    use_side_info: bool

def normalize(array: np.ndarray) -> np.ndarray:
    """Normalize an array by its maximum value (with numerical stability)."""
    return array / (array.max() + 1e-12)

def plot_results(input_data: np.ndarray, result: np.ndarray, gt: np.ndarray, config: Configuration):
    """
    Plot CHM, DTM, zoomed CHM, and profile images for the input, diffusion result, and ground truth.
    
    Assumes that `createCHM` returns a 3-element tuple (or list) of arrays.
    """
    # Compute CHM for each dataset
    input_chm = createCHM(input_data)
    result_chm = createCHM(result)
    gt_chm = createCHM(gt)

    # Constants for plotting (adjust as needed)
    zoom_x = (16 * 6) * 3
    zoom_y = (32 * 3) * 3
    profile_line = 100
    upper_limit = -1  # as in your code (was set to 140 then overwritten)
    lower_limit = 0
    dtm_cmap = 'copper'

    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    fig.subplots_adjust(hspace=0, wspace=0)

    # Row 0: Input data
    axs[0, 0].imshow(input_chm[0], interpolation='nearest', aspect=2)
    axs[0, 1].imshow(input_chm[1], interpolation='nearest', aspect=2, cmap=dtm_cmap)
    axs[0, 1].imshow(input_chm[2], interpolation='nearest', aspect=2, cmap='Greys', alpha=0.35)
    axs[0, 2].imshow(input_chm[0][zoom_x // 6:zoom_x // 6 + 16, zoom_y // 3:zoom_y // 3 + 32],
                      interpolation='nearest', aspect=2)
    axs[0, 3].imshow(
        np.flip(input_data[lower_limit // 2:upper_limit // 2, profile_line // 6, -200 // 3:], axis=0),
        cmap='gray_r', interpolation='nearest', aspect=0.25
    )

    # Row 1: Diffusion result
    axs[1, 0].imshow(result_chm[0], interpolation='nearest', vmin=gt_chm[0].min(), vmax=gt_chm[0].max())
    axs[1, 1].imshow(result_chm[1], interpolation='nearest', cmap=dtm_cmap)
    axs[1, 1].imshow(result_chm[2], interpolation='nearest', cmap='Greys', alpha=0.35)
    axs[1, 2].imshow(
        result_chm[0][zoom_x // config.resolution:zoom_x // config.resolution + (96 // config.resolution),
                      zoom_y // config.resolution:zoom_y // config.resolution + (96 // config.resolution)],
        interpolation='nearest'
    )
    axs[1, 3].imshow(
        np.flip(result[lower_limit:upper_limit, profile_line // config.resolution, -200 // config.resolution:], axis=0),
        cmap='gray_r', interpolation='nearest', aspect=1 / config.resolution
    )

    # Row 2: Ground truth
    axs[2, 0].imshow(gt_chm[0], interpolation='nearest')
    axs[2, 1].imshow(gt_chm[1], interpolation='nearest', cmap=dtm_cmap)
    axs[2, 1].imshow(gt_chm[2], interpolation='nearest', cmap='Greys', alpha=0.35)
    axs[2, 2].imshow(
        gt_chm[0][zoom_x // config.resolution:zoom_x // config.resolution + (96 // config.resolution),
                  zoom_y // config.resolution:zoom_y // config.resolution + (96 // config.resolution)],
        interpolation='nearest'
    )
    axs[2, 3].imshow(
        np.flip(gt[lower_limit:upper_limit, profile_line // config.resolution, -200 // config.resolution:], axis=0),
        cmap='gray_r', interpolation='nearest', aspect=1 / config.resolution
    )

    # Titles and labels
    axs[0, 0].set_title('CHM')
    axs[0, 1].set_title('DTM')
    axs[0, 2].set_title('CHM Zoom')
    axs[0, 3].set_title('Profile')
    axs[0, 0].set_ylabel('Input')
    axs[1, 0].set_ylabel('Diffusion (Side Info)' if config.use_side_info else 'Diffusion')
    axs[2, 0].set_ylabel(f'Ground Truth ({config.resolution}m)')

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    save_path = f'./results/Diffusion_{config.resolution}_use_side_{config.use_side_info}.png'
    plt.savefig(save_path)
    plt.show()

def main():
    # === Parse arguments and create configuration ===
    parser = argparse.ArgumentParser(
        description="Diffusion model sampling with optional side information."
    )
    parser.add_argument('--resolution', type=int, default=3, help="Generated image resolution")
    parser.add_argument('--batch_size', type=int, default=64, help="Training batch size")
    parser.add_argument('--train_num_steps', type=int, default=100000, help="Total training steps")
    parser.add_argument('--use_side_info', type=int, default=0, help="Use side information (1 to enable)")
    args = parser.parse_args()

    config = Configuration(
        resolution=args.resolution,
        train_batch_size=args.batch_size,
        train_num_steps=args.train_num_steps,
        use_side_info=(args.use_side_info == 1)
    )

    # === Load data from testCubes folder ===
    # For ground truth and input data, the filename reflects the resolution.
    # For photo data, we use the file at resolution 1.
    input_path = f"testCubes/input{config.resolution}.npy"
    gt_path = f"testCubes/gt{config.resolution}.npy"
    photo_path = "testCubes/photo.npy"

    input_data = np.load(input_path)
    gt = np.load(gt_path)
    photo = np.load(photo_path)

    # === Preprocessing ===
    crop_val = 72  # equivalent to "valor" in your original code

    gt = normalize(gt)
    input_data = normalize(input_data)
    photo = normalize(photo)

    # Compute ratios based on shapes
    _, h_gt, w_gt = gt.shape
    _, h_lr, w_lr = input_data.shape
    ratio1 = h_gt // h_lr
    ratio2 = w_gt // w_lr

    # Crop arrays (using your original cropping logic)
    gt = gt[:, :8 * crop_val, :8 * crop_val]
    input_data = input_data[:, : (8 * crop_val) // 6, : (8 * crop_val) // 3]
    photo = photo[:, :8 * crop_val, :8 * crop_val]

    # Swap axes for correct orientation
    input_data = np.swapaxes(input_data, 1, 2)
    gt = np.swapaxes(gt, 1, 2)
    photo = np.swapaxes(photo, 1, 2)

    # Process side information from photo data
    photo_data = createCHM(photo)[0]
    photo_data = photo_data / (photo_data.max() + 1e-16)

    # Convert input and photo data to torch tensors and add noise to input
    input_tensor = torch.tensor(input_data, device='cuda', dtype=torch.float32).unsqueeze(0)
    input_tensor += torch.randn_like(input_tensor) * 0.075

    photo_tensor = torch.tensor(photo_data * 2 - 1, device='cuda', dtype=torch.float32)\
        .unsqueeze(0).unsqueeze(0)
    if not config.use_side_info:
        photo_tensor = None

    # === Initialize model, diffusion, and trainer ===
    model = Unet(
        dim=128,
        dim_mults=(8, 16, 16, 16),
        flash_attn=True,
        channels=257 if config.use_side_info else 256,
        out_dim=128
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=96 // config.resolution,
        sampling_timesteps=50  # number of sampling timesteps (using DDIM for faster inference)
    )

    trainer = Trainer(
        diffusion,
        train_batch_size=config.train_batch_size,
        train_lr=8e-5,
        train_num_steps=config.train_num_steps,
        gradient_accumulate_every=64 // config.train_batch_size,
        ema_decay=0.995,
        amp=True,
        resolution=config.resolution,
        use_side_info=config.use_side_info
    )

    trainer.load(config.resolution, config.use_side_info)

    # === Sampling / Diffusion Process ===
    if config.use_side_info:
        forward_imaging = ForwardImaging(config.resolution, trainer.device)
        output = torch.randn(
            1, 128, input_tensor.shape[-2] * ratio2, input_tensor.shape[-1] * ratio1,
            device='cuda'
        )
        pbar = tqdm(reversed(range(1000)), total=1000)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t in pbar:
                output = output.requires_grad_()
                output_pred, x_start = trainer.ema.ema_model.p_sample(
                    output, t, lowRes=(input_tensor * 2 - 1), photo=photo_tensor
                )
                x_start = (x_start + 1) * 0.5

                distri, _ = forward_imaging.forward_imaging(x_start)
                loss = -distri.log_prob(input_tensor[0].transpose(-1, -3) / 1.2).mean()

                lr_step = 1
                grad = torch.autograd.grad(loss, output)[0]
                output = output_pred - lr_step * grad
                output = output.detach()
                pbar.set_postfix({'loss': loss.item()})
            output = (output.transpose(-1, -2) + 1) / 2
    else:
        with torch.no_grad():
            output = trainer.ema.ema_model.p_sample_loop(
                (1, 128, input_tensor.shape[-2] * ratio2, input_tensor.shape[-1] * ratio1),
                lowRes=(input_tensor * 2 - 1),
                photo=photo_tensor
            ).transpose(-1, -2)

    # === Post-process and prepare for plotting ===
    result = output[0].cpu().detach().numpy()
    result[result < 0.04] = 0

    # Crop the input and ground truth for plotting (using constant 66 as in the original code)
    input_plot = input_tensor.cpu().detach().squeeze(0).transpose(1, 2).numpy()
    input_plot = input_plot[:, : (8 * 66) // 6, : (8 * 66) // 3]
    gt_plot = gt[:, :8 * 66 // config.resolution, :8 * 66 // config.resolution]
    result_plot = result[:, :8 * 66 // config.resolution, :8 * 66 // config.resolution]

    # === Plot and save results ===
    plot_results(input_plot, result_plot, gt_plot, config)

if __name__ == '__main__':
    main()
