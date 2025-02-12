import numpy as np
from typing import Tuple


def apply_kernel(dtm: np.ndarray, center: Tuple[int, int], k_size: int) -> np.ndarray:
    """
    Extract a subsection of the digital terrain model (DTM) centered at the given point 
    with the specified kernel size.

    The function ensures that the extracted window does not exceed the boundaries of the DTM.

    Args:
        dtm (np.ndarray): 2D array representing the digital terrain model.
        center (Tuple[int, int]): (row, column) coordinates of the center point.
        k_size (int): Size of the kernel (assumed to be an odd integer).

    Returns:
        np.ndarray: The extracted subsection of the DTM.
    """
    half_size = k_size // 2
    x_center, y_center = center

    # Determine the start and end indices, ensuring they are within the DTM's bounds.
    x_start = max(x_center - half_size, 0)
    y_start = max(y_center - half_size, 0)
    x_end = min(x_center + half_size + 1, dtm.shape[0])
    y_end = min(y_center + half_size + 1, dtm.shape[1])

    return dtm[x_start:x_end, y_start:y_end]


def adaptive_dtm_filter(dtm: np.ndarray, k_size: int = 7) -> np.ndarray:
    """
    Filter noisy pixels in the digital terrain model (DTM) based on local height statistics.

    For each pixel, if its value exceeds the median of its local neighborhood, it is 
    replaced with the mean of the neighboring values that are below or equal to the median.

    Args:
        dtm (np.ndarray): 2D array representing the digital terrain model.
        k_size (int, optional): Size of the local window used for filtering. Defaults to 7.

    Returns:
        np.ndarray: The filtered digital terrain model.
    """
    filtered_dtm = dtm.copy()
    rows, cols = dtm.shape

    for i in range(rows):
        for j in range(cols):
            # Extract the local window around the current pixel.
            window = apply_kernel(filtered_dtm, (i, j), k_size).flatten()

            if window.size > 0:
                median_val = np.percentile(window, 50)
                if filtered_dtm[i, j] > median_val:
                    # Get all window values less than or equal to the median.
                    lower_values = window[window <= median_val]
                    if lower_values.size > 0:
                        filtered_dtm[i, j] = lower_values.mean()

    return filtered_dtm


def hillshade(elevation: np.ndarray, azimuth: float = 90, angle_altitude: float = 60) -> np.ndarray:
    """
    Compute hillshade values from a digital elevation model (DEM) to simulate terrain shading.

    Args:
        elevation (np.ndarray): 2D array representing elevation data.
        azimuth (float, optional): Light source azimuth in degrees. Defaults to 90.
        angle_altitude (float, optional): Light source altitude angle in degrees. Defaults to 60.

    Returns:
        np.ndarray: Hillshade values scaled to the range [0, 255].
    """
    # Invert the azimuth to match hillshade conventions.
    azimuth = 360.0 - azimuth

    # Compute gradients in the x and y directions.
    x, y = np.gradient(elevation.astype(float))
    # Calculate slope: the angle between the surface and the horizontal.
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x**2 + y**2))
    # Calculate aspect: the direction of the steepest slope.
    aspect = np.arctan2(-x, y)

    # Convert angles from degrees to radians.
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)

    # Calculate hillshade using the standard formula.
    shaded = (np.sin(altitude_rad) * np.sin(slope) +
              np.cos(altitude_rad) * np.cos(slope) *
              np.cos(azimuth_rad - np.pi / 2.0 - aspect))

    # Scale the result to the 0-255 range.
    hillshade_values = 255 * (shaded + 1) / 2
    return hillshade_values


def calcSurf(cubo: np.ndarray, porcentaje: float) -> np.ndarray:
    """
    Calculate the surface index from a volumetric data cube by finding the first layer where 
    the cumulative normalized sum exceeds a given threshold.

    Args:
        cubo (np.ndarray): 3D array where the first axis represents layers (e.g., height or depth).
        porcentaje (float): Threshold (between 0 and 1) to determine the surface.

    Returns:
        np.ndarray: 2D array representing the surface index for each (row, column) position.
    """
    # Compute the cumulative sum along the first axis.
    cumulative = np.cumsum(cubo, axis=0)
    # Get the maximum cumulative value for each spatial position.
    max_values = np.max(cumulative, axis=0)
    # Prevent division by zero.
    max_values[max_values == 0] = 1
    # Normalize the cumulative sum.
    cumulative_normalized = cumulative / max_values

    # Identify where the normalized cumulative sum exceeds the threshold.
    above_threshold = cumulative_normalized > porcentaje
    # For each spatial location, find the first layer index where the threshold is exceeded.
    surface = np.argmax(above_threshold, axis=0)

    return surface


def createCHM(cubo: np.ndarray, porcentaje: float = 0.98) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Canopy Height Model (CHM) from a volumetric data cube.

    The function computes the Digital Terrain Model (DTM) and Digital Elevation Model (DEM)
    using different thresholds and then calculates the CHM as the difference (DEM - DTM). 
    It also computes a hillshade of the filtered DTM.

    Args:
        cubo (np.ndarray): 3D volumetric data array where the first axis is the height or depth.
        porcentaje (float, optional): Threshold for determining the DEM. Defaults to 0.98.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - chm: The Canopy Height Model (DEM - filtered DTM).
            - dtm: The filtered Digital Terrain Model.
            - dtm_hillshade: Hillshade representation of the filtered DTM.
            - dem: The Digital Elevation Model.
    """
    # Calculate the Digital Terrain Model (DTM) using a low threshold.
    dtm = calcSurf(cubo, 0.02)
    # Apply adaptive filtering to mitigate noise in the DTM.
    dtm_filtered = adaptive_dtm_filter(dtm)
    # Calculate the Digital Elevation Model (DEM) using the provided threshold.
    dem = calcSurf(cubo, porcentaje)
    # Compute the Canopy Height Model (CHM) as the difference between DEM and DTM.
    chm = dem - dtm_filtered

    return chm, dtm_filtered, hillshade(dtm_filtered), dem
