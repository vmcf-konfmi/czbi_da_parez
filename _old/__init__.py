# __init__.py

"""
This package contains modules for feature analysis, image processing, and data visualization.
"""

# Import key functions or classes for easier access
from package.feature_analysis import (
    load_data,
    assign_columns,
    group_and_pivot,
    drop_columns,
    correlation_analysis,
    tsne_clustering,
    umap_clustering,
    pca_analysis,
    pca_visualization,
    run_feature_analysis,
)

from package.image_channel_analysis import (
    version,
    image_info,
    normalize,
    plot_normalized_channels,
    measure_brightness_variability,
    count_nuclei_under_membrane,
    percentage_nuclei_pixels,
    measure_membrane_pixel_count,
    measure_membrane_thickness,
    save_label_mask,
    create_mask_from_labels,
    image_th_ch1,
    image_th_ch2,
    quartiles,
    measure_image,
    summarize_features,
    process_imag_old,
    process_image,
    detect_boundary,
    detect_boundary_wider,
    gradient_analysis,
    remove_large_objects,
    analyze_bright_spots,
    prep_for_membrane_analysis,
    analyze_membrane_continuity,
    process_images_in_subfolders,
    format_data_to_dataframe,
)

from package.quality import (
    check_image_paths,
    print_directory_tree,
    image_info,
)

from utilities.ipynb_helper import (
    convert_ipynb_to_py,
)

from utilities.ipynb_todo_check import (
    find_todos_in_notebook,
    ipynb_todo_check,
)