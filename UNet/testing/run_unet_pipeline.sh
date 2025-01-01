#!/bin/bash
# example usage: ./run_unet_pipeline.sh /path/to/model_checkpoint.pth /path/to/prob_output /path/to/combined_output 0.5 /path/to/threshold_output
# Check if the correct number of arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <model_checkpoint> <prob_tiles_output> <combined_output> <threshold> <threshold_output>"
    echo "Example:"
    echo "./run_unet_pipeline.sh /path/to/model_checkpoint.pth /path/to/prob_output /path/to/combined_output 0.5 /path/to/threshold_output"
    exit 1
fi

# Assign input arguments to variables
MODEL_CHECKPOINT=$1       # Model checkpoint path
PROB_TILES_OUTPUT=$2      # Probability tiles output path
COMBINED_OUTPUT=$3        # Combined output directory path
THRESHOLD=$4              # Threshold value (e.g., 0.5)
THRESHOLD_OUTPUT=$5       # Output path for thresholded results

# Define constant input paths
DSM_TILES="/Users/nirgut/Documents/Master_Files/UNet_files/Train_Test_Data/128_overlap/DSM_tiles/test_not_empty_normalized"
MASK_TILES="/Users/nirgut/Documents/Master_Files/UNet_files/Train_Test_Data/128_overlap/Mask_tiles/test_not_empty"
SHAPEFILE="/Users/nirgut/Documents/Master_Files/UNet_files/Base_Rasters/base_shp/selected_test_empty_tile_shapes.shp"
MASK_TEST_GROUPS="/Users/nirgut/Documents/Master_Files/UNet_files/Base_Rasters/Mask_test_groups"

# Set logging level
LOG_LEVEL="INFO"

# Print the input parameters for confirmation
echo "Running with the following inputs:"
echo "Model Checkpoint: $MODEL_CHECKPOINT"
echo "Probability Tiles Output: $PROB_TILES_OUTPUT"
echo "Combined Output: $COMBINED_OUTPUT"
echo "Threshold: $THRESHOLD"
echo "Threshold Output: $THRESHOLD_OUTPUT"

# Step 1: Run UNet testing script
echo "Running UNet testing script..."
#./UNet_test_new.py "$DSM_TILES" "$MASK_TILES" "$MODEL_CHECKPOINT" -O "$PROB_TILES_OUTPUT" -l "$LOG_LEVEL"

# Step 2: Group and combine probability tiles
echo "Grouping and combining probability tiles..."
python3 group_and_combine_prob_tiles.py "$SHAPEFILE" "$PROB_TILES_OUTPUT" "$COMBINED_OUTPUT"

# Step 3: Apply threshold and test results
# create in combined output directory a combined_groups directory
mkdir -p "$COMBINED_OUTPUT/combined_groups"
echo "Applying threshold and testing results..."
./apply_threshold_and_test.py "${COMBINED_OUTPUT}/combined_groups/" "$MASK_TEST_GROUPS" "$THRESHOLD" "$THRESHOLD_OUTPUT" -l "$LOG_LEVEL"
#./apply_threshold_and_test.py "${COMBINED_OUTPUT}/" "$MASK_TEST_GROUPS" "$THRESHOLD" "$THRESHOLD_OUTPUT" -l "$LOG_LEVEL"

echo "Pipeline execution completed successfully!"

