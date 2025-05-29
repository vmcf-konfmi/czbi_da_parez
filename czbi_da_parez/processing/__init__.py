# __init__.py for czbi_da_parez.processing package

# Re-export all public functions for flat import style
from .image_channel_analysis import *
from .quality import *
from .feature_summary import *
from .membrane_analysis import *
from .batch_processing import *

# Optionally, define __all__ for explicit export
from .image_channel_analysis import (
    image_info, normalize, plot_normalized_channels, image_th_ch1, image_th_ch2
)
from .quality import (
    measure_brightness_variability, count_nuclei_under_membrane, percentage_nuclei_pixels
)
from .feature_summary import (
    measure_membrane_pixel_count, measure_membrane_thickness, measure_image, summarize_features
)
from .membrane_analysis import (
    save_label_mask, create_mask_from_labels, detect_boundary, detect_boundary_wider, gradient_analysis, analyze_bright_spots, prep_for_membrane_analysis, analyze_membrane_continuity
)
from .batch_processing import (
    process_image, process_imag_old, process_images_in_subfolders, format_data_to_dataframe
)

__all__ = [
    # image_channel_analysis
    "image_info", "normalize", "plot_normalized_channels", "image_th_ch1", "image_th_ch2",
    # quality
    "measure_brightness_variability", "count_nuclei_under_membrane", "percentage_nuclei_pixels",
    # feature_summary
    "measure_membrane_pixel_count", "measure_membrane_thickness", "measure_image", "summarize_features",
    # membrane_analysis
    "save_label_mask", "create_mask_from_labels", "detect_boundary", "detect_boundary_wider", "gradient_analysis", "analyze_bright_spots", "prep_for_membrane_analysis", "analyze_membrane_continuity",
    # batch_processing
    "process_image", "process_imag_old", "process_images_in_subfolders", "format_data_to_dataframe"
]
