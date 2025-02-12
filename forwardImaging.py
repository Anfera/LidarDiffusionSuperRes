import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union


class ForwardImaging:
    """
    A class to perform forward imaging simulation using a Gaussian kernel convolution.

    The module upsamples the input high-resolution image along the spatial dimensions and
    then applies a 3D convolution using a precomputed Gaussian kernel. Finally, it constructs
    a multinomial distribution based on the convolved output.

    Attributes:
        resolution (int): Upsampling factor for the spatial dimensions.
        device (Union[str, torch.device]): Device for tensor computations.
        vertical_super_res (bool): Flag indicating if vertical super-resolution is enabled.
        kernel_size (int): Size of the Gaussian kernel (kernel_size x kernel_size).
        sigma (float): Standard deviation for the Gaussian kernel.
        padding (list): Padding to be applied during convolution.
        stride (list): Stride values for the convolution.
        kernel (torch.Tensor): Precomputed Gaussian kernel tensor.
    """

    def __init__(
        self,
        resolution: int,
        device: Union[str, torch.device] = "cpu",
        vertical_super_res: bool = False
    ) -> None:
        """
        Initialize the ForwardImaging module.

        Args:
            resolution (int): Upsampling factor for the height and width.
            device (Union[str, torch.device], optional): Device to perform computations.
                Defaults to "cpu".
            vertical_super_res (bool, optional): Whether to use vertical super-resolution.
                Defaults to False.
        """
        self.resolution = resolution
        self.device = device
        self.vertical_super_res = vertical_super_res

        # Set convolution parameters.
        # For vertical super-resolution, use a stride of 2 along the depth dimension.
        self.padding = [0, 4, 5]  # [pad_depth, pad_height, pad_width]
        self.stride = [2, 3, 6] if vertical_super_res else [1, 3, 6]

        self.kernel_size = 11
        self.sigma = 6 / 2.35

        # Create a 2D Gaussian kernel and replicate it along the depth if needed.
        kernel_np = self.gaussian_kernel(self.kernel_size, self.sigma)
        # Create a tensor from the numpy array on the specified device.
        kernel_tensor = torch.tensor(kernel_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Replicate the kernel for vertical super-resolution if required.
        num_channels = 2 if vertical_super_res else 1
        self.kernel = torch.ones(num_channels, self.kernel_size, self.kernel_size, device=self.device) * kernel_tensor
        # After this, self.kernel has shape (num_channels, kernel_size, kernel_size).

    def gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """
        Generate a normalized 2D Gaussian kernel.

        Args:
            size (int): The size of the kernel (size x size).
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            np.ndarray: Normalized 2D Gaussian kernel.
        """
        # Create a coordinate grid centered at zero.
        ax = np.arange(size) - (size - 1) / 2.0
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def forward_imaging(self, high_res: torch.Tensor, number_photons: int = 20) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Process the high-resolution image by upsampling and convolving it with the Gaussian kernel,
        then construct a multinomial distribution from the aggregated output.

        Args:
            high_res (torch.Tensor): High-resolution image tensor with shape (N, D, H, W),
                where N is the batch size, D is depth, and H and W are spatial dimensions.

        Returns:
            Tuple[torch.distributions.Distribution, torch.Tensor]:
                - A multinomial distribution with a fixed total count (e.g., 20) and probabilities
                  derived from the aggregated (convolved) tensor.
                - The aggregated tensor obtained after the convolution.
        """
        # Add a channel dimension so that high_res has shape (N, 1, D, H, W).
        high_res = high_res.unsqueeze(1)

        # Unpack the input dimensions.
        batch_size, channels, depth, height, width = high_res.shape

        # Upsample the spatial dimensions (height and width) by the resolution factor.
        # The depth remains unchanged.
        new_size = (depth, height * self.resolution, width * self.resolution)
        high_res = F.interpolate(high_res, size=new_size, mode="nearest")

        # Prepare the convolution kernel.
        # Reshape self.kernel to have shape:
        # (out_channels=1, in_channels=1, kernel_depth, kernel_height, kernel_width)
        # For vertical super-resolution, kernel_depth equals num_channels (2); otherwise, it is 1.
        conv_kernel = self.kernel.unsqueeze(0).unsqueeze(0)

        # Apply 3D convolution.
        aggregated = F.conv3d(high_res, conv_kernel, stride=self.stride, padding=self.padding)

        # Remove the channel dimension.
        aggregated = aggregated.squeeze(1)

        # Adjust tensor dimensions for the multinomial distribution.
        # Swap the third-to-last and last dimensions and add a small constant for stability.
        probabilities = aggregated.transpose(-3, -1) + 1e-12

        # Create a multinomial distribution with a fixed total count (e.g., 20).
        distribution = torch.distributions.Multinomial(total_count=number_photons, probs=probabilities)

        return distribution, aggregated
