"""https://arxiv.org/abs/2501.15890"""

import sys
import os
from typing import Dict, List
import cv2
import numpy as np


def calculate_MSG(img: np.ndarray) -> float:
    """
    Calculate Multi-Scale Sobel Gradient (MSG) score for an image.

    Parameters:
        img: Input image in BGR format (OpenCV's default format)

    Returns:
        Combined color gradient score across multiple scales (0-1 range)
    """
    # Normalization and initialization
    img_normalized = img.astype(np.float32) / 255.0
    msg_score = 0.0

    # Multi-scale configuration
    scales = [1, 2, 4, 8]
    scale_weights = [0.4, 0.3, 0.2, 0.1]

    for scale, weight in zip(scales, scale_weights):
        # Downsample image
        new_size = (img.shape[1] // scale, img.shape[0] // scale)
        scaled_img = cv2.resize(img_normalized, new_size)

        # Calculate channel gradients
        channel_gradients = []
        for channel in range(3):  # Process BGR channels
            # Calculate spatial gradients
            grad_x = cv2.Sobel(scaled_img[:, :, channel], cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(scaled_img[:, :, channel], cv2.CV_32F, 0, 1, ksize=3)

            # Compute gradient magnitude
            magnitude = np.hypot(grad_x, grad_y)
            channel_gradients.append(np.mean(magnitude))

        # Aggregate scale contribution
        scale_gradient = np.mean(channel_gradients)
        msg_score += weight * scale_gradient

    return msg_score


def calculate_canny_edge_density(img: np.ndarray) -> float:
    """
    Calculate edge density using Canny edge detection.

    Parameters:
        img: Input image in BGR or grayscale format

    Returns:
        Ratio of edge pixels to total pixels (0-1 range)
    """
    # Convert to grayscale if necessary
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Noise reduction and edge detection
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    median_val = np.median(blurred)

    # Adaptive threshold calculation
    lower = int(max(0, 0.67 * median_val))
    upper = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, lower, upper)

    # Calculate density metrics
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    return edge_pixels / total_pixels


def calculate_multiscale_unique_colors(
    img: np.ndarray, bits_per_channel: int = 5
) -> Dict:
    """
    Analyze color diversity across multiple scales with quantization.

    Parameters:
        img: Input image in BGR format
        bits_per_channel: Color resolution bits (1-8)

    Returns:
        Dictionary containing color diversity metrics
    """
    # Configuration
    scales = [1, 2, 4, 8]
    scale_weights = [0.4, 0.3, 0.2, 0.1]
    color_shift = 8 - bits_per_channel
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scale_results = []
    for scale in scales:
        # Downsample with area interpolation
        new_size = (img.shape[1] // scale, img.shape[0] // scale)
        scaled_img = cv2.resize(rgb_img, new_size, interpolation=cv2.INTER_AREA)

        # Quantize color space
        quantized = (scaled_img.astype(np.int32) >> color_shift) << color_shift

        # Create unique color identifiers
        flat_colors = quantized.reshape(-1, 3)
        color_ids = (
            flat_colors[:, 0].astype(np.int64) * (2**16)
            + flat_colors[:, 1].astype(np.int64) * (2**8)
            + flat_colors[:, 2]
        )

        # Calculate color metrics
        unique_colors = np.unique(color_ids)
        color_count = len(unique_colors)
        max_colors = (2**bits_per_channel) ** 3

        # Normalized metrics
        log_norm = np.log10(color_count + 1) / np.log10(max_colors + 1)

        scale_results.append(
            {
                "scale": scale,
                "unique_colors": color_count,
                "dimensions": scaled_img.shape[:2],
            }
        )

    # Weighted combination
    color_counts = [r["unique_colors"] for r in scale_results]
    weighted_sum = sum(c * w for c, w in zip(color_counts, scale_weights))

    return {
        "multiscale_score": round(weighted_sum, 2),
        "color_resolution": bits_per_channel,
    }


def main():
    """Command-line interface for image quality assessment."""
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <image_path>")
        sys.exit(1)

    try:
        img = cv2.imread(sys.argv[1])
        if img is None:
            raise FileNotFoundError(f"Could not load image at {sys.argv[1]}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Calculate all metrics
    msg_score = calculate_MSG(img)
    edge_density = calculate_canny_edge_density(img)
    color_metrics = calculate_multiscale_unique_colors(img)

    # Print results in a structured format
    print("\nVisual Features:")
    print("----------------------------------")
    print(f"1. MSG Score: {msg_score:.4f}")
    print(f"2. Edge Density (Canny): {edge_density:.4f}")

    print("3. Colorfulness Analysis:")
    print(f"   - MUC Score: {color_metrics['multiscale_score']}")
    print(
        f"   - Color Resolution (# of bits to preserve for each channel): {color_metrics['color_resolution']}"
    )
    print("----------------------------------")


if __name__ == "__main__":
    main()
