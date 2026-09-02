# Sinkhole Mapping with U-Net

Automated detection and mapping of Dead Sea sinkholes from high-resolution remote sensing data (e.g., LiDAR DSM) using an optimized U-Net deep learning architecture.

This project is part of the **Geological Survey of Israel's Dead Sea Research**, aiming to streamline geomorphic hazard monitoring through scalable semantic segmentation pipelines.

---

## Key Updates & Enhancements

- **Windowed Streaming Inference:** Large rasters are processed in streaming windows directly to disk, eliminating Out-Of-Memory (OOM) crashes on multi-gigabyte datasets.
- **Edge-Tile Boundary Handling:** Accurate window slicing, padding, and cropping prevent coordinate offset and border clipping artifacts.
- **Native GeoTIFF & Compression:** Direct writing via GDAL's `GTiff` driver using LZW compression to prevent data bloat.
- **GPU Acceleration & Batching:** Native CUDA support with configurable batch processing and classification probability thresholds.
- **Automated Vectorization:** End-to-end extraction from raw raster probability outputs to cleaned ESRI Shapefiles (`.shp`).

---

## Project Structure

```text
Sinkholes_ml/
├── UNet/
│   ├── data/                 # Raw dataset preparation and training patch generation
│   │   └── Creating_Training_Data.py
│   ├── functions/            # Core inference and GIS post-processing utilities
│   │   ├── Applying_UNet_oren.py   # Optimized inference script (windowed streaming)
│   │   ├── SinkTools2.py           # Polygonization, geometry filtering, and cleaning
│   │   ├── python_starter_code.py
│   │   └── validate_tiles.py
│   ├── notebooks/            # Exploratory analysis and prototyping notebooks
│   │   ├── Applying_UNet.ipynb
│   │   ├── Creating_Training_Data.ipynb
│   │   ├── UNet_test.ipynb
│   │   ├── UNet_train.ipynb
│   │   └── commands.ipynb
│   ├── testing/              # Model validation, threshold exploration, and metric evaluation
│   │   ├── UNet_test_new.py
│   │   ├── apply_threshold_and_test.py
│   │   ├── combine_tiles.sh
│   │   ├── group_and_combine_prob_tiles.py
│   │   └── run_unet_pipeline.sh
│   ├── training/             # Network definitions, loss functions, and training loops
│   │   ├── UNet_train.py
│   │   ├── data.py
│   │   ├── loss.py
│   │   ├── model.py
│   │   └── utils.py
│   └── txt_files/            # Documentation notes and workflow guides
│       └── UNet_test_new_guide.txt
├── .gitignore
└── README.md
