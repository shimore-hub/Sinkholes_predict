#!/bin/bash

# usage example: ./combine_tiles.sh /path/to/tiles_dir /path/to/output_dir output_name

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 tiles_dir output_dir output_name"
    exit 1
fi

# Input arguments
TILES_DIR=$1
OUTPUT_DIR=$2
OUTPUT_NAME=$3

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Step 1: Create VRT Mosaic
echo "Creating VRT mosaic..."
gdalbuildvrt "$OUTPUT_DIR/${OUTPUT_NAME}.vrt" "$TILES_DIR"/*.tif
if [[ $? -ne 0 ]]; then
    echo "Error: gdalbuildvrt failed."
    exit 1
fi

# Step 2: Edit VRT file for summing tiles
echo "Editing VRT file for summing tiles..."
sed -i '' '/<VRTRasterBand /c\
<VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">\
    <PixelFunctionType>sum</PixelFunctionType>\
    <SourceTransferType>Float32</SourceTransferType>\
    <NoDataValue>0</NoDataValue>' "$OUTPUT_DIR/${OUTPUT_NAME}.vrt"
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to edit VRT file."
    exit 1
fi

# Step 3: Convert VRT to GeoTIFF
echo "Converting VRT to GeoTIFF..."
gdal_translate -of GTiff "$OUTPUT_DIR/${OUTPUT_NAME}.vrt" "$OUTPUT_DIR/combined_groups/${OUTPUT_NAME}.tif"
if [[ $? -ne 0 ]]; then
    echo "Error: gdal_translate failed."
    exit 1
fi

echo "Mosaic successfully created at $OUTPUT_DIR/${OUTPUT_NAME}.tif"

