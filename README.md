# czbi-da-parez

A Python package for data processing, image channel analysis, and feature analysis, designed for reproducible research and extensible workflows.

## Importing Functions and Usage

You can import functions and classes from `czbi_da_parez` in two main ways:

### 1. Flat Import (Recommended)
All public functions from the processing submodules are re-exported at the top level. This means you can import any function directly from the main package or the `processing` subpackage:

```python
# Import any function directly from the package (flat import)
from czbi_da_parez import analyze_bright_spots, process_image, measure_image

# Or from the processing subpackage
from czbi_da_parez.processing import analyze_bright_spots, process_image, measure_image
```

### 2. Submodule Import
You can also import functions or classes from their specific submodules if you prefer:

```python
from czbi_da_parez.processing.membrane_analysis import analyze_bright_spots
from czbi_da_parez.processing.batch_processing import process_image
from czbi_da_parez.processing.feature_summary import measure_image
```

### Example Usage
```python
from czbi_da_parez import process_image
summary_df_ch1, summary_df_ch2, specific_df = process_image('path/to/image.tif')
```

Both import styles are supported. The flat import style is convenient for scripts and notebooks, while submodule imports are useful for clarity in larger projects.

## Features
- **Processing**: Quality checks, image info extraction, and channel/mask processing.
- **Analysis**: Feature extraction, clustering, dimensionality reduction, and statistical analysis.
- **Utilities**: Notebook conversion, TODO checks, and helper functions.

## Installation

Clone the repository and install dependencies:

```sh
pip install -e .
```

Or using poetry (if preferred):

```sh
poetry install
```

## Requirements
- Python >= 3.11
- See `pyproject.toml` for all dependencies (opencv-python, scikit-image, pandas, numpy, stardist, tensorflow, watermark, umap-learn, hdbscan, seaborn, scikit-learn, nbformat, nbconvert)

## Package Structure

```
czbi_da_parez/
    processing/
        quality.py
        image_channel_analysis.py
    analysis/
        feature_analysis.py
    utilities/
        ipynb_helper.py
        ipynb_todo_check.py
    __init__.py
```

- **processing**: Data loading, cleaning, transformation, and image channel/mask analysis.
- **analysis**: Feature extraction, clustering, and statistical analysis.
- **utilities**: Helper functions for notebooks and scripts.

## Usage Example

```python
from czbi_da_parez.processing.quality import QualityChecker
from czbi_da_parez.processing.image_channel_analysis import ImageChannelAnalyzer
from czbi_da_parez.analysis.feature_analysis import FeatureAnalyzer

# Quality check
QualityChecker.check_image_paths(df, 'image_path')

# Image info
size, channels, px_size = ImageChannelAnalyzer.image_info('path/to/image.tif')

# Feature analysis
df = FeatureAnalyzer.load_data('data.csv')
df = FeatureAnalyzer.assign_columns(df)
```

## Documentation

Sphinx documentation is available in the `docs/` folder. To build the docs and ensure all module/class/function docstrings are included:

1. Make sure Sphinx and its extensions are installed:

```powershell
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme
```

2. (Optional but recommended) Regenerate the API documentation stubs using sphinx-apidoc. From the project root, run:

```powershell
sphinx-apidoc -o docs czbi_da_parez
```

This will create or update `docs/czbi_da_parez.*.rst` files that include all modules, classes, and functions with their docstrings.

3. Build the HTML documentation:

```powershell
sphinx-build -b html docs docs/_build/html
```

4. Open `docs/_build/html/index.html` in your browser to view the documentation.

**Tips for full docstring coverage:**
- Ensure all your public classes, methods, and functions have Sphinx-style docstrings (triple quotes, with parameter and return descriptions).
- In `docs/conf.py`, make sure you have these extensions enabled:
  - `sphinx.ext.autodoc`
  - `sphinx.ext.napoleon` (for Google/Numpy style docstrings)
- If you want deeper module documentation, use the `:members:`, `:undoc-members:`, and `:show-inheritance:` options in your `.rst` files, e.g.:

  ```rst
  .. automodule:: czbi_da_parez.processing.quality
      :members:
      :undoc-members:
      :show-inheritance:
      :inherited-members:
  ```

- You can edit `docs/modules.rst` to add more modules or adjust the depth.

## Notebooks

Example Jupyter notebooks are in the `colab/` folder. Use the utilities to convert or check them as needed.

## License

MIT License

## Getting Started with Dev Container

This project includes a pre-configured VS Code Dev Container for a reproducible development environment. To get started:

1. **Open in VS Code**: Open the project folder in [Visual Studio Code](https://code.visualstudio.com/).
2. **Reopen in Container**: Press <kbd>F1</kbd> (or <kbd>Ctrl+Shift+P</kbd>) and run the command:
   
   ```
   Dev Containers: Open Folder in Container
   ```
   
   VS Code will build and launch the development environment inside a container, installing all dependencies automatically.
3. **Start Coding**: Once the container is ready, you can run, test, and develop the package as described below.

This ensures a consistent environment across all contributors and avoids dependency issues on your local machine.