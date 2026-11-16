# Stone Characterisation and Spatial Analysis

This folder contains the scripts responsible for analysing the physical characteristics and spatial distribution of detected stones.

The main purpose of this stage is to estimate the real-world dimensions and physical properties of individual stones, prepare their spatial information, identify duplicate detections using DBSCAN clustering, and visualize the resulting stone distributions.

The analysis is divided into two main components: stone characterisation and spatial analysis.

## Folder Structure

### 1. `stone_characterisation/`

**Stone size, classification, volume and mass estimation**

Contains scripts for estimating the physical properties of detected stones using image-based measurements and geometric assumptions.

- **`Stone size estimation and classification.py`**: Converts pixel-based detection areas into real-world measurements using Ground Sampling Distance (GSD). Estimates effective stone footprints and equivalent circular diameters, then assigns size classes based on configurable diameter thresholds.
- **`Stone volume and mass estimation.py`**: Estimates stone volume using a simplified flat-ellipsoid model, assuming thickness is proportional to the equivalent stone diameter. Calculates approximate stone mass using an assumed material density.

The resulting attributes include estimated diameter, size class, volume and mass.

### 2. `spatial_analysis/`

**Spatial data preparation, clustering and visualization**

Contains scripts for preparing georeferenced stone detections, grouping spatially close detections and evaluating the resulting clustering distributions.

#### 2.1 `data_preparation/`

Contains:

- **`Preparation of stone detection data for DBSCAN clustering.py`**: Extracts spatial coordinates and relevant detection attributes from the original detection dataset and generates a simplified CSV file suitable for DBSCAN clustering.
- **`Stone detection filtering by real-world size class.py`**: Filters detection records according to their assigned real-world size categories, allowing selected stone size groups to be analysed separately.

#### 2.2 `clustering/`

Contains:

- **`DBSCAN parameter range testing and sensitivity analysis.py`**: Runs DBSCAN using multiple spatial distance thresholds (eps), evaluates their effects on detection grouping and noise identification, and saves stone-level results and summary statistics for each tested parameter value.
- **`DBSCAN-based stone detection deduplication.py`**: Groups spatially close detections using DBSCAN in UTM coordinates and aggregates each cluster into a single representative stone record.

The clustering process converts detection-level records into stone-level records by calculating representative spatial coordinates and aggregating available stone measurements.

#### 2.3 `visualization/`

Contains:

- **`DBSCAN clustering result visualization.py`**: Generates spatial scatter plots of stone locations obtained using different DBSCAN distance thresholds, allowing visual comparison of the resulting stone distributions.
- **`Before-and-after DBSCAN spatial comparison.py`**: Generates overlay scatter plots comparing the original stone detections with the deduplicated stone locations obtained through DBSCAN clustering.

These visualizations support the interpretation and evaluation of the spatial clustering results.

## Overall Workflow

**Georeferenced Detection Data → Spatial Data Preparation → Stone Size Estimation and Classification → Optional Size Filtering → DBSCAN Parameter Testing → Final DBSCAN Deduplication → Spatial Visualization → Stone-Level Results**

After clustering, the resulting stone-level diameter information can be used for further volume and mass estimation.

## Important Notes

- The spatial analysis scripts require valid UTM coordinates (`x_utm` and `y_utm`) expressed in meters.
- The stone size estimation script requires bounding-box area (`area_px`) and Ground Sampling Distance (`gsd_m_per_px`).
- Size filtering is optional and can be used to analyse specific stone size categories separately.
- DBSCAN parameter testing helps investigate suitable distance thresholds (`eps`). It does not automatically select the optimal value.
- The final deduplication script should use the selected DBSCAN parameters.
- Stone volume and mass values are estimates based on the assumed geometric model, thickness ratio and material density.
- The `prepare_for_dbscan.py` script retains a defined list of columns. Make sure the diameter and size-class attributes are preserved in the data passed to DBSCAN when they are needed in the aggregated stone-level results.
- Update the input and output paths and parameter settings according to the analysis requirements before execution.