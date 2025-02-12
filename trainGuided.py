import logging
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def main() -> None:
    """
    Main function to initialize the model, diffusion process, and trainer,
    then load a checkpoint (if available) and begin training.
    """
    # Define the resolution factor.
    resolution: int = 3

    # Initialize the UNet model with specified parameters.
    model = Unet(
        dim=128,
        dim_mults=(8, 16, 16, 16),
        flash_attn=True,
        channels=128
    )

    # Set up the Gaussian Diffusion process.
    diffusion = GaussianDiffusion(
        model,
        image_size=96 // resolution,
        timesteps=1000,           # Total diffusion steps.
        sampling_timesteps=250    # Sampling timesteps (using DDIM for faster inference).
    )

    # Configure the trainer with training hyperparameters.
    trainer = Trainer(
        diffusion,
        train_batch_size=8,
        train_lr=8e-5,
        train_num_steps=700000,         # Total training steps.
        gradient_accumulate_every=8,    # Gradient accumulation steps.
        ema_decay=0.995,                # Exponential moving average decay.
        amp=True,                       # Enable mixed precision training.
        resolution=resolution
    )

    # Attempt to load a checkpoint; if not found, log a warning and start from scratch.
    try:
        trainer.load(0)
    except Exception as error:
        logging.warning("No checkpoint found, training from scratch. Error: %s", error)

    # Start the training process.
    trainer.train()


if __name__ == "__main__":
    # Configure logging to display informational messages.
    logging.basicConfig(level=logging.INFO)
    main()
