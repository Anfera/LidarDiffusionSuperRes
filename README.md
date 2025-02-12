# Super-Resolved 3D Satellite Lidar Imaging of Earth Via Generative Diffusion Models

This repository contains code for generating HyperHeight DataCubes (HHDCs) from low-resolution input data using diffusion models. The project offers many inference approaches:

**1. Diffusion with Posterior Distribution:**

- **File:** `inferenceDistribution.py`
- **Description:** This approach uses a pre-trained diffusion model to generate high-resolution CHMs. The diffusion process is guided by low-resolution input data, which helps to improve the accuracy and detail of the generated CHMs.
- **Configuration:**
    - `--resolution`: Generated image resolution (default: 3).
    - `--use_side_info`: Flag to use side information (1 to use, 0 otherwise) (default: 0).
    - `--load_cube`: Flag to load a pre-generated cube (1 to load, 0 otherwise) (default: 0).


**2. Guided Diffusion:**

- **File:** `inferenceGuided.py`
- **Description:** This approach incorporates a forward imaging model into the diffusion process. The forward imaging model simulates the process of acquiring low-resolution data from high-resolution HHDCs. This allows the diffusion model to learn the relationship between the low-resolution input and the high-resolution output more effectively.
- **Configuration:**
    - `--resolution`: Generated image resolution (default: 3).
    - `--batch_size`: Training batch size (default: 64).
    - `--train_num_steps`: Total training steps (default: 100000).
    - `--use_side_info`: Flag to use side information (1 to enable) (default: 0).

**Additional Files:**

- `canopyPlots.py`: Contains functions for generating CHMs and visualizing results.
- `forwardImaging.py`: Implements the forward imaging model.
- `trainGuided.py`: Provides code for training the guided diffusion model.
- `denoising_diffusion_pytorch`: A folder containing the implementation of the diffusion models and related utilities.

**Requirements:**

- See `requirements.txt` for a list of required Python packages.

**Usage:**

1. Install the required packages: `pip install -r requirements.txt`
2. Configure the desired inference approach by setting the appropriate arguments in the corresponding file.
3. Run the inference file: `python inferenceGuided.py` or `python inferenceDistribution.py`

**Results:**

The generated CHMs and visualizations will be saved in the `results` folder.

**Note:**

- The project requires pre-trained model weights, which are not included in this repository.
- The `testCubes` folder should contain the low-resolution input data and high-resolution ground truth data for testing. It is necessary to decompress them.
- The `Dataset` folder should contain the training data for the diffusion models.

**Acknowledgments:**

- This project is based on the `denoising_diffusion_pytorch` library. https://github.com/lucidrains/denoising-diffusion-pytorch