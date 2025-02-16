import logging
import argparse
from dataclasses import dataclass
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


@dataclass
class Configuration:
    resolution: int
    train_batch_size: int
    train_num_steps: int
    use_side_info: bool


def parse_args() -> Configuration:
    """
    Parse command-line arguments and return a Configuration instance.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_num_steps', type=int, default=700000)
    parser.add_argument('--use_side_info', type=int, default=0)
    args = parser.parse_args()

    logging.info(
        f"resolution: {args.resolution}, batch_size: {args.batch_size}, "
        f"train_num_steps: {args.train_num_steps}, use_side_info: {args.use_side_info}"
    )

    return Configuration(
        resolution=args.resolution,
        train_batch_size=args.batch_size,
        train_num_steps=args.train_num_steps,
        use_side_info=(args.use_side_info == 1)
    )


def main() -> None:
    """
    Main function to initialize configuration, model, diffusion process, and trainer,
    then load a checkpoint (if available) and begin training.
    """
    # Parse command-line arguments to get the configuration.
    config = parse_args()

    # Initialize the UNet model with the specified parameters.
    model = Unet(
        dim=160,
        dim_mults=(8, 16, 16, 16),
        flash_attn=True,
        channels=257 if config.use_side_info else 256,
        out_dim=128
    )

    # Set up the Gaussian Diffusion process.
    diffusion = GaussianDiffusion(
        model,
        image_size=96 // config.resolution,
        sampling_timesteps=250  # Using DDIM for faster inference.
    )

    # Configure the Trainer with training hyperparameters.
    trainer = Trainer(
        diffusion,
        train_batch_size=config.train_batch_size,
        train_lr=8e-5,
        train_num_steps=config.train_num_steps,  # Total training steps.
        gradient_accumulate_every=64 // config.train_batch_size,
        ema_decay=0.995,
        amp=True,
        resolution=config.resolution,
        use_side_info=config.use_side_info
    )

    # Attempt to load a checkpoint; if not found, log a warning and train from scratch.
    try:
        trainer.load(config.resolution, config.use_side_info)
    except Exception as error:
        logging.warning("No checkpoint found, training from scratch. Error: %s", error)

    # Start the training process.
    trainer.train()


if __name__ == "__main__":
    # Configure logging to display informational messages.
    logging.basicConfig(level=logging.INFO)
    main()
