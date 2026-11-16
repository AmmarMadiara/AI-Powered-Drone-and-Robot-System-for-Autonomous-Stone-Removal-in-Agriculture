# Image Detection Pipeline

This folder contains the scripts responsible for image preprocessing, YOLO dataset preparation, model-assisted annotation and inference post-processing in the stone-detection project.

The main purpose of this stage is to prepare large original images for object detection, organize the training dataset, support manual verification of model-generated predictions, and reconstruct patch-level detections back onto the original images.

## Folder Structure

### 1. `image_preprocessing/`

**Image patch generation and quality filtering**

Contains scripts for dividing large original images into smaller overlapping patches suitable for YOLO object detection.

The two available approaches are:

- **`patche_images_for_yolo_inference.py`**: Generates full-coverage overlapping image patches and records their original image coordinates and inner-window metadata. This metadata is required for reconstructing predictions on the original images.
- **`patch generation and quality filtering.py`**: Generates overlapping patches, calculates image-quality and texture metrics, and retains selected patches based on configurable quality thresholds.

These scripts provide alternative patch-generation approaches depending on the intended application.

### 2. `dataset_preparation/`

**YOLO dataset preparation and splitting**

Contains:

- **`prepare_yolo_dataset.py`**: Combines positive images with optional negative images, divides the dataset into training, validation and testing subsets, and generates the corresponding YOLO `data.yaml` configuration file.

The script expects existing image-label pairs for positive images and creates empty annotation files for negative images.

### 3. `annotation/`

**YOLO-assisted annotation and manual verification**

Contains:

- **`YOLO-to-CVAT annotation conversion.py`**: Converts YOLO prediction labels into CVAT 1.1 XML format, allowing model-generated stone detections to be imported into CVAT for visual inspection and manual correction.

This stage supports a human-in-the-loop annotation workflow:

1. Run YOLO inference on image patches.
2. Convert the predicted labels into CVAT XML.
3. Import the annotations and corresponding images into CVAT.
4. Inspect and correct the predicted bounding boxes.
5. Add missed stones and remove false detections.
6. Export the corrected annotations in YOLO format for subsequent training.

The corrected annotations can be incorporated into the training dataset to support further model improvement.

### 4. `inference/`

**YOLO prediction reconstruction and post-processing**

Contains:

- **`reconstruct_yolo_detections.py`**: Maps YOLO predictions from individual image patches back to their corresponding original image coordinates.

The script uses the patch metadata generated during full-coverage image preprocessing to:
- Convert patch-level bounding boxes into original image coordinates.
- Remove duplicate detections using the inner-window rule.
- Optionally apply Non-Maximum Suppression (NMS).
- Generate detection overlays on original images.
- Export reconstructed detections as YOLO TXT labels and CSV files.

## Overall Workflow

**Original Images → Image Patch Generation → Dataset Preparation → YOLO Model Training → YOLO Inference → CVAT Annotation Verification → Detection Reconstruction**

The CVAT annotation workflow provides an iterative feedback process in which manually corrected predictions can be used to improve subsequent YOLO training.

Detection reconstruction is a separate post-inference operation that produces georeferencing-ready detection results on the original images.

## Important Notes

- The full-coverage patch generation approach should be used when the inner-window metadata is required for detection reconstruction.
- The quality-filtered patch generation approach is intended to reduce uninformative patches and does not generate the same reconstruction metadata.
- Dataset preparation expects corresponding image and annotation files to already exist.
- Update the input and output paths in the scripts according to the local project directory before execution.
- CVAT correction and YOLO retraining are separate operations from the annotation conversion script itself.