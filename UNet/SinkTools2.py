import os
import numpy as np
import math
# import sys
# import pandas as pd
from osgeo import gdal, gdalconst, osr, ogr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
import time
import pickle
import scipy.ndimage as ndimage
from scipy.stats import zscore
from skimage import io, morphology
import geopandas as gpd
from shapely.geometry import Polygon
from functools import reduce
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import rasterio
from rasterio import features
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from pathlib import Path
from rasterio.mask import mask
import fiona
from matplotlib import pyplot as plt
import sys
import subprocess
import pyproj as Proj
import tqdm
import random
import torch
import shutil


def clip_raster_by_raster(src_ras, ref_ras, output_path):
    # Takes a raster file and clips it with the extents of another raster file.
    # src_ras- the raster we want to clip. ref_ras- the raster with the extents we want.

    # Open the reference raster and get its extent
    ref_ds = gdal.Open(ref_ras, gdalconst.GA_ReadOnly)
    ulx, xres, _, uly, _, yres = ref_ds.GetGeoTransform()
    lrx = ulx + (ref_ds.RasterXSize * xres)
    lry = uly + (ref_ds.RasterYSize * yres)

    # Build the clipping window from the reference raster's extent
    clip_window = (ulx, uly, lrx, lry)

    # Open the source raster and clip it using the clipping window
    src_ds = gdal.Open(src_ras, gdalconst.GA_ReadOnly)
    gdal.Translate(output_path, src_ds, projWin=clip_window, format="GTiff")
    del src_ds


def get_proj(raster):
    rast_src = gdal.Open(raster)
    # rast_src = LiT.reproject_dataset(rast_src)
    # rast_src.FlushCache()
    geotransform = rast_src.GetGeoTransform()
    projection = rast_src.GetProjection()
    return geotransform, projection


def fix_nodata(ref_raster, new_raster, output):
    """
    This function takes a reference raster and a new raster and replaces the nodata values in the new raster with the
    :param ref_raster:
    :param new_raster:
    :param output:
    :return:
    """

    ref = gdal.Open(ref_raster, gdalconst.GA_ReadOnly)
    ref_array = ref.ReadAsArray().astype(np.int32)
    print(ref_array.shape)

    # Open new_raster and get its projection and geotransform
    ras = gdal.Open(new_raster, gdalconst.GA_ReadOnly)
    ras_proj = ras.GetProjection()
    ras_geotrans = ras.GetGeoTransform()
    ras_array = ras.ReadAsArray().astype(np.int32)
    print(ras_array.shape)
    # Replace values in matrix A based on matrix B
    ras_array[ref_array == -32767] = -32767
    # ras_array[0, 0] = -32767
    # plc_array[plc_array == -9999] = -32767
    # ras_array[ras_array <= -10] = -32767
    # ras_array[ras_array >= 1000] = -32767
    print('saving')
    # Create output raster
    save_array_as_tiff(ras_array, output, ras_geotrans, ras_proj)


def filter_small_obj(raster, output, num):
    """

    :param raster:
    :param output:
    :param num:
    :return:
    """
    # Load the input image
    img = io.imread(raster)
    geotran, proj = get_proj(raster)
    # Convert the data type to bool
    img_bool = img.astype(bool)

    # Set the minimum size for the objects to be removed
    min_size = num

    # Remove small objects from the image
    filtered_img = morphology.remove_small_objects(img_bool, min_size=min_size, connectivity=2)
    print(type(filtered_img))
    # make sure that in the values the original raster had -32767 value, also in the filtered raster it will have
    # -32767 value

    filtered_img[img_bool == -32767] = -32767
    # Save the filtered image to a file
    save_array_as_tiff(filtered_img, output, geotran, proj)
    # # Save the filtered image to a file
    # io.imsave(output, filtered_img)
    #
    # geotrans, proj = get_proj(raster)
    # # Create a new raster file and write the filtered image to it
    # rast = gdal.GetDriverByName('GTiff').Create(output, filtered_img.shape[1], filtered_img.shape[0], 1, gdal.GDT_Byte)
    # rast.SetGeoTransform(geotrans)
    # rast.SetProjection(proj)
    # rast.GetRasterBand(1).WriteArray(filtered_img.astype(np.uint8))
    # rast.FlushCache()


def pred_array2raster(original_raster, pred_array, output_path):
    # takes an array (probability or prediction) with 3 columns (2 coordinates & 1 value) and transforms to tif file
    pred = np.load(pred_array)
    raster = gdal.Open(original_raster)
    original_array = raster.ReadAsArray()
    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()
    print(1)
    # get dimensions/sizes of original raster
    rows = raster.RasterYSize
    cols = raster.RasterXSize
    print(f'rows: {rows}, cols: {cols}')
    ras_array = np.zeros([rows, cols])
    print(2)
    # if the value of a ixel in the original raster is -32767 (nodata) then the value in the new raster will be -32767
    ras_array[original_array == -32767] = -32767

    # Loop through the rows of the smaller array
    for i in range(pred.shape[0]):
        col = int(pred[i, 0])
        row = int(pred[i, 1])
        val = pred[i, 2]
        ras_array[row, col] = val
    print(3)
    save_array_as_tiff(ras_array, output_path, geotransform, projection)


def save_array_as_tiff(array, output_path, geotransform, projection, nodata_val=-32767):
    """
    Saves a numpy array as a tiff file
    :param array: numpy array in the size of the raster
    :param output_path: path for the output tiff file
    :param geotransform:
    :param projection:
    :param nodata_val:
    :return:
    """
    print(geotransform)
    print(projection)
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    print(rows, cols)
    print(type(rows))
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).SetNoDataValue(nodata_val)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    del dataset


def choose_random(file, output, rows_num):
    array = np.load(file)
    num_rows = array.shape[0]
    rows_to_select = np.random.choice(num_rows, size=rows_num, replace=False)

    # select the rows using NumPy indexing
    selected_rows = array[rows_to_select, :]

    # save the selected rows to a .npy file
    np.save(output, selected_rows)


def split_binary(array, output0, output1):
    data = np.load(array)

    sinkholes_0 = data[data[:, -1] == 0]
    sinkholes_1 = data[data[:, -1] == 1]

    print(sinkholes_0.shape)
    print(sinkholes_1.shape)

    shape1 = sinkholes_1.shape[0]

    np.save(output1, sinkholes_1)
    np.save(output0, sinkholes_0)
    # delete from memory
    del sinkholes_0
    del sinkholes_1
    del data
    return shape1


def separate_col(array_path, output_dir):
    # Load the numpy array from file
    data = np.load(array_path)

    # Select the last 8 columns of the array
    last_8_columns = data[:, -8:]

    # Split the last 8 columns into separate arrays
    columns = np.split(last_8_columns, 8, axis=1)
    col_names = ['DEM', 'Slope', 'Aspect', 'PLC', 'PRC', 'MBI', 'TWI', 'Sinkholes']
    # Save each column to a separate npy file
    for i, col in enumerate(columns):
        np.save(str(output_dir) + '/MSH_' + str(col_names[i]) + '.npy', col)


def raster_to_table(rasters, output_path):
    # Load each raster, reproject, and read as a numpy array
    arrays = []
    for raster in rasters:
        print(f'starts reading {raster}')
        rast_src = gdal.Open(raster, gdalconst.GA_ReadOnly)
        # rast_src = gdal.Translate('', rast_src, format='VRT')
        # rast_src = gdal.Warp('', rast_src, format='VRT', dstSRS='EPSG:4326')
        rast_array = rast_src.ReadAsArray()
        print(rast_array.shape)
        arrays.append(rast_array)
        print(f'finished reading {raster}')

    # Get the shape of the arrays
    shape = arrays[0].shape

    # Add x and y position columns
    x_coords, y_coords = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x_coords = x_coords.flatten().reshape(-1, 1)
    y_coords = y_coords.flatten().reshape(-1, 1)
    print('finished meshgrid')
    # Reshape each array into a 1D array and concatenate along the second axis
    table = np.concatenate([x_coords, y_coords], axis=1)
    count = 0
    for array in arrays:
        count += 1
        flat_array = array.flatten().reshape(-1, 1)
        table = np.concatenate([table, flat_array], axis=1)
        print(f'finished reshaping array {count}')
    # remove array from memory
    del arrays
    np.save(output_path, table)
    print(f'Table saved to: {output_path}')
    # remove table from memory
    del table


def raster_to_table2(rasters, output_path):
    # Load each raster, reproject, and read as a numpy array
    arrays = []
    for raster in rasters:
        print(f'starts reading {raster}')
        rast_src = gdal.Open(raster, gdalconst.GA_ReadOnly)
        # rast_src = gdal.Translate('', rast_src, format='VRT')
        # rast_src = gdal.Warp('', rast_src, format='VRT', dstSRS='EPSG:4326')
        rast_array = rast_src.ReadAsArray()
        print(rast_array.shape)
        arrays.append(rast_array)
        print(f'finished reading {raster}')

    # Get the shape of the arrays
    shape = arrays[0].shape

    # Add x and y position columns
    x_coords, y_coords = raster2coordinates(rasters[0])
    print('finished meshgrid')
    print(f'x_coords shape: {x_coords.shape}')
    print(f'y_coords shape: {y_coords.shape}')
    x_coords = x_coords.flatten().reshape(-1, 1)
    y_coords = y_coords.flatten().reshape(-1, 1)
    print(f'x_coords shape: {x_coords.shape}')
    print(f'y_coords shape: {y_coords.shape}')
    # Reshape each array into a 1D array and concatenate along the second axis
    table = np.concatenate([x_coords, y_coords], axis=1)

    count = 0
    for array in arrays:
        count += 1
        flat_array = array.flatten().reshape(-1, 1)
        table = np.concatenate([table, flat_array], axis=1)
        print(f'finished reshaping array {count}')
    # remove array from memory
    del arrays
    np.save(output_path, table)
    print(f'Table saved to: {output_path}')
    # remove table from memory
    del table


def raster2coordinates(raster):
    # Open the raster and get its properties
    rast_src = gdal.Open(raster)
    geotransform = rast_src.GetGeoTransform()
    projection = rast_src.GetProjection()
    rows = rast_src.RasterYSize
    cols = rast_src.RasterXSize

    # Calculate the x and y coordinates using the geotransform information
    x_origin, pixel_width, _, y_origin, _, pixel_height = geotransform
    x_coords = x_origin + np.arange(cols) * pixel_width
    y_coords = y_origin + np.arange(rows) * pixel_height

    # Create meshgrid for all pixel coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Reshape the coordinates into column vectors
    x_coords = xx.reshape(-1, 1)
    y_coords = yy.reshape(-1, 1)

    return x_coords, y_coords


def rasters2ones_and_rand0(aoi, rasters, binary_raster, dem_raster, output_folder, n_0files=1, mult_rand=1):
    """
    This function takes a list of rasters and converts them to a table with the x and y coordinates and the values of
    the rasters. It also takes a binary raster and a dem raster and adds the values of the binary raster to the table.
    from these values it makes a table for only binary 1 values, and "n_0files" number of tables for binary 0 values.
    it uses the regular dem raster to pick random 0 values only from relevant areas.
    :param aoi: the path to the aoi
    :param rasters: list of raster paths
    :param binary_raster:
    :param dem_raster:
    :param output_folder: the path for the output folder
    :param n_0files: the number of 0 files to create
    :return:
    """
    start = time.time()
    binary = gdal.Open(binary_raster, gdalconst.GA_ReadOnly)
    binary_array = binary.ReadAsArray()
    print('finished reading binary raster')
    del binary
    dem = gdal.Open(dem_raster, gdalconst.GA_ReadOnly)
    dem_array = dem.ReadAsArray()
    print('finished reading dem raster')
    del dem
    # Get the shape of the arrays
    shape = dem_array.shape

    # Add x and y position columns
    x_coords, y_coords = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    print('finished meshgrid')
    # value_1 finds all the places in binary array that are 1 and puts them on a list with the adjacent coordinates
    # from x_coords and y_coords
    value_1_coor = (np.array([x_coords[binary_array == 1], y_coords[binary_array == 1]])).T
    print('finished value_1_coor')
    # print amount of 1's in binary raster
    print(f'amount of 1s in binary raster: {value_1_coor.shape[0]}')
    # extract the values from dem_raster that are in the coordinates of value_1_coor
    value_1 = dem_array[value_1_coor[:, 1], value_1_coor[:, 0]]
    value_1 = np.column_stack((value_1_coor, value_1))
    # remove dem_array from value_1
    value_1 = np.delete(value_1, 2, 1)

    for raster in rasters:
        raster_gdal = gdal.Open(raster, gdalconst.GA_ReadOnly)
        raster_array = raster_gdal.ReadAsArray()
        print(f'finished reading {raster}')
        del raster_gdal
        # extract the values from raster that are in the coordinates of value_1_coor and add as column to value_1
        value_1 = np.column_stack((value_1, raster_array[value_1_coor[:, 1], value_1_coor[:, 0]]))
        print(f'finished adding {raster} to value_1')
        del raster_array
    value_1 = np.column_stack((value_1, np.ones(value_1.shape[0])))
    np.save(output_folder + str(aoi) + '_value_1.npy', value_1)
    print('finished saving value_1')

    # choose random values that has value 0 in binary_raster and is between the values -434 and -370 in dem_raster
    value_0_coor = (np.array([x_coords[(binary_array == 0) & (dem_array < -370) & (dem_array > -434)],
                              y_coords[(binary_array == 0) & (dem_array < -370) & (dem_array > -434)]]).T)
    print('finished value_0_coor')
    # choose random values  in the size of value_1_coor
    for i in range(n_0files):
        value_0_coor = value_0_coor[np.random.choice(value_0_coor.shape[0], value_1_coor.shape[0]*mult_rand, replace=False), :]
        print(f'finished random choice number {i + 1}')
        # extract the values from dem_raster that are in the coordinates of value_0_coor
        value_0 = dem_array[value_0_coor[:, 1], value_0_coor[:, 0]]
        print('finished value_0')

        # add the coordinates from the left
        value_0 = np.column_stack((value_0_coor, value_0))
        # add the values from the other rasters to the value_1 and value_0 arrays
        print('starting to add rasters to value_1 and value_0')
        # remove dem_array from value_0
        value_0 = np.delete(value_0, 2, 1)

        for raster in rasters:
            raster_gdal = gdal.Open(raster, gdalconst.GA_ReadOnly)
            raster_array = raster_gdal.ReadAsArray()
            print(f'finished reading {raster}')
            del raster_gdal
            # extract the values from raster that are in the coordinates of value_0_coor and add as column to value_0
            value_0 = np.column_stack((value_0, raster_array[value_0_coor[:, 1], value_0_coor[:, 0]]))
            print(f'finished adding {raster} to value_0')
            del raster_array
        # add the binary raster to the value_1 and value_0 arrays from the right
        value_0 = np.column_stack((value_0, np.zeros(value_0.shape[0])))
        # save tables from value_1 and value_0 as npy files
        np.save(output_folder + str(aoi) + '_value_0_rand_#' + str(i + 1) + '.npy', value_0)
    del dem_array
    print('finished saving value_0_rand')
    print(f'Tables saved to: {output_folder + str(aoi)} + _value_1.npy and _value_0_rand.npy')
    # Create and write details to a text file
    details_file_path = os.path.join(output_folder, str(aoi) + '_details.txt')

    with open(details_file_path, 'w') as details_file:
        details_file.write("Details about the saved files:\n")
        details_file.write(f"value_1.npy:\n")
        details_file.write(f"  - Shape: {value_1.shape}\n")
        details_file.write(f"  - Columns: x_coord, y_coord, {str([os.path.basename(r) for r in rasters])}, "
                           f"binary\n")
        details_file.write(f"number of random files: {n_0files}\n")
        details_file.write(f"Other saved files:\n")
        details_file.write(f"  - Depending on the input rasters, additional npy files may be present.\n")

    print(f'Details saved to: {details_file_path}')
    end = time.time()
    print(f'Total time: {end - start}')


def filter_table_by_std(table_path, output_path, std_num):
    # Load the table from the specified path
    data = np.load(table_path, mmap_mode='r')
    data_len = data.shape[1]
    # Apply filtering conditions for specific columns
    for col in range(2, data_len - 1):
        col_data = data[:, col]
        col_mean = np.mean(col_data)
        col_std = np.std(col_data)
        threshold = std_num * col_std

        # Filter the data points beyond the threshold
        data = data[np.abs(col_data - col_mean) <= threshold]
        print(f'{col} finished')

    # Save the filtered table as a .npy file
    np.save(output_path, data)
    # delete table from memory
    del data


def filter_table_by_std2(table_path, output_path, std_num):
    # Load the table from the specified path
    data = np.load(table_path, mmap_mode='r')
    data_len = data.shape[1]
    # remove rows in which column 3 equals to -9999
    data = data[data[:, 3] != -9999]
    data = data[data[:, 2] != -32767]
    filtered_data = []

    # Apply filtering conditions for specific columns
    for col in range(2, data_len - 1):
        col_data = data[:, col]
        print(col_data.shape)
        col_mean = np.mean(col_data)
        print(col_mean)
        col_std = np.std(col_data)
        print(col_std)
        threshold = std_num * col_std
        print(threshold)

        # Filter the data points beyond the threshold
        mask = np.abs(col_data - col_mean) <= threshold
        filtered_col_data = col_data[mask]
        print(filtered_col_data.shape)
        filtered_data.append(filtered_col_data)
        print(f'{col} finished')

    # Concatenate the filtered columns horizontally
    filtered_data = np.column_stack(filtered_data)

    # Save the filtered table as a .npy file
    np.save(output_path, filtered_data)

    # Delete variables from memory
    del data, filtered_data


def replace_nodata_value(raster_path, new_value, output_path=None, nodata_value=None):
    """
    Replace the nodal value in a raster with a new value.

    Parameters:
        raster_path (str): The path to the input raster file.
        new_value (int or float): The value to replace the nodal value with.
        nodata_value (int or float): The nodal value to be replaced. If not provided,
                                     the nodal value of the input raster will be used.
        output_path (str): The path to save the output raster file. If not provided,
                            the input raster will be updated in place.

    Returns:
        None
    """

    # Load the input raster
    with gdal.Open(raster_path, gdal.GA_Update) as raster:

        # Get the nodal value if not provided
        if nodata_value is None:
            nodata_value = raster.GetRasterBand(1).GetNoDataValue()

        # Read the raster as a numpy array
        array = raster.ReadAsArray()

        # Replace the nodal `value with the new value
        array[array == nodata_value] = new_value

        # Save the updated raster if an output path is provided
        if output_path:
            driver = gdal.GetDriverByName('GTiff')
            out_raster = driver.CreateCopy(output_path, raster)
            out_raster.GetRasterBand(1).WriteArray(array)
            out_raster.FlushCache()
            # out_raster = None
        else:
            raster.GetRasterBand(1).WriteArray(array)
            raster.FlushCache()

    # The dataset is automatically closed after the 'with' block
    return


def zscore_raster(raster_path, output_path):
    print(f'Processing {raster_path}')
    start_time = time.time()
    # Open the raster file
    data = gdal.Open(raster_path)
    # get projection and geotransform
    proj = data.GetProjection()
    gt = data.GetGeoTransform()
    # Read the raster data as a NumPy array
    band = data.GetRasterBand(1)  # Assuming you want to read the first band
    raster_array = band.ReadAsArray()
    print(raster_array.shape)
    # Get the nodata value
    nodata_value = band.GetNoDataValue()
    print('nodata value:', nodata_value)

    # Calculate the z-score of each value in the array
    if nodata_value is not None:
        raster_array_without_nodata = np.ma.masked_equal(raster_array, nodata_value)
        print('raster array without nodata shape:', raster_array_without_nodata.shape)
        zscore_array = zscore(raster_array_without_nodata, axis=None)
    else:
        zscore_array = zscore(raster_array, axis=None)
        print('zscore array shape:', zscore_array.shape)

    print('calculating zscore finished')
    z_score_time = time.time()
    print(f'time used for {raster_path}:', z_score_time - start_time)
    # Save the z-score array to a new raster file
    save_array_as_tiff(zscore_array, output_path, gt, proj)
    saving_time = time.time()
    print('saving time:', saving_time - z_score_time)


def divide_array(array_path, output_dir, num_parts):
    # Load the numpy array from file
    data = np.load(array_path)

    # Get the number of rows in the array
    num_rows = data.shape[0]

    # Calculate the number of rows in each part
    part_rows = num_rows // num_parts

    # Split the array into num_parts parts
    for i in range(num_parts):
        if i < num_parts - 1:
            part = data[i * part_rows:(i + 1) * part_rows]
        else:
            part = data[i * part_rows:]
        # Save the part to a separate npy file
        np.save(str(output_dir) + '/part_' + str(i) + '.npy', part)
        # delete part from memory
        del part


def combine_arrays(array_list, output_path):
    # Load the first array from file
    data = np.load(array_list[0])
    print('first array loaded')

    # Loop through the remaining arrays and concatenate them
    for array in array_list[1:]:
        data = np.concatenate([data, np.load(array)], axis=0)
        # delete array from memory
        print('array concatenated')
        del array

    # Save the combined array to file
    np.save(output_path, data)
    # delete data from memory
    del data


def scaling_all_data(train_path_list, test_path_list, train_output_list, test_output_list):
    # initialize the scaler
    scaler = StandardScaler()
    print(3)
    # loop through the training data files and partial fit the scaler
    for table in train_path_list:
        data = np.load(table)
        # Only scale columns 2 to 8
        print(data[:, -1:].shape)
        scaler.partial_fit(data[:, 2:-1])
        print(data[:, 2:-1].shape)
        # delete loaded train data from memory
        del data
        print(table + ' fitted')

    for i, table in enumerate(train_path_list):
        data = np.load(table)
        # Only scale columns 2 to 8
        scaled_cols = scaler.transform(data[:, 2:-1])
        scaled_data = np.hstack([data[:, :2], scaled_cols, data[:, -1:]])
        np.save(train_output_list[i], scaled_data)
        del data

    # Save the scaler object
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    for i, table in enumerate(test_path_list):
        data = np.load(table)
        # Only scale columns 2 to 8
        scaled_cols = scaler.transform(data[:, 2:-1])
        scaled_data = np.hstack([data[:, :2], scaled_cols, data[:, -1:]])
        np.save(test_output_list[i], scaled_data)
        del data


def elevation_per(raster, radius, output_path):
    start_time = time.time()  # Start the timer
    radius = radius * 2  # Multiplying because of the pixel res which is 0.5 m
    rast_src = gdal.Open(raster)
    # Get the geotransform and projection of the raster
    geotransform, projection = get_proj(raster)
    # Read the raster data into an array
    array = rast_src.ReadAsArray()
    # Get the dimensions of the array
    length = range(array.shape[0])  # Rows
    width = range(array.shape[1])  # Columns
    print(f'width: {width}')
    print(f'length: {length}')
    # Create an array to store the elevation values
    elev = np.zeros([array.shape[0], array.shape[1]])
    # Iterate over each row
    for row in length:
        if row in (1000, 5000, 10000, 15000, 20000, 12000, 22000, 25000, 30000, 35000, 37000):
            print(row)
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
        # Iterate over each column
        for col in width:
            x = array[row, col]  # Get the value at the current row and column
            # Extract a sub-matrix centered around the current pixel
            mat = array[max(0, row - radius):min(array.shape[0], row + radius + 1),
                        max(0, col - radius):min(array.shape[1], col + radius + 1)]
            # Count the number of elements in the sub-matrix that are smaller than x
            boolarr = mat < x
            small_vals = boolarr.sum()
            # Calculate the elevation percentage by dividing the count by the total non-zero elements
            elev[row, col] = small_vals / np.count_nonzero(mat)
    # Save the elevation array as a TIFF file
    save_array_as_tiff(elev, output_path, geotransform, projection)


def coeff2pred(intercept, coeff, test_array, threshold, prob_out, pred_out):
    test = np.load(test_array)
    X_test = test[:, 2:-1]  # all columns except the last one
    y_test = test[:, -1]  # the last column only
    coefficients = np.load(coeff)
    intercept = np.load(intercept)

    lr_model = LogisticRegression()
    lr_model.coef_ = coefficients
    lr_model.intercept_ = intercept
    lr_model.classes_ = np.array([0, 1])  # Set the classes manually
    probs = lr_model.predict_proba(X_test)
    print('predicted')

    # The first column of the probs array will contain the predicted probabilities for the negative class (0),
    # and the second column will contain the predicted probabilities for the positive class (1).
    coordinates = test[:, 0:2]
    final_prob_array = np.concatenate((coordinates, probs[:, 1].reshape(-1, 1)), axis=1)
    np.save(prob_out, final_prob_array)
    print('saved probabilitys')
    y_pred = np.where(probs[:, 1] >= threshold, 1, 0)
    print('threshold finished')

    # accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    sinkhole_acc = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    not_sinkhole_acc = conf_mat[0, 0] / (conf_mat[0, 1] + conf_mat[0, 0])
    FPR = conf_mat[0, 1] / (conf_mat[1, 1] + conf_mat[0, 1])

    print(5)
    # print("Accuracy:", accuracy)
    print("Sinkhole Accuracy:", sinkhole_acc)
    print("Non-Sinkhole Accuracy:", not_sinkhole_acc)
    print("False Positive Rate:", FPR)
    print("Confusion matrix:\n", conf_mat)

    y_pred = y_pred.reshape(-1, 1)
    print(coordinates.shape)
    print(y_pred.shape)
    final_pred_array = np.concatenate((coordinates, y_pred), axis=1)
    np.save(pred_out, final_pred_array)


def auto_tif2bin(rast_list, output_folder, file_name):
    """
    this function takes a list of geomorphic features rasters and turns it to 2 tables:
    a table with all the sinkhole pixels data (1)
    a table of random non-sinkhole pixels in the size of the sinkhole pixels data.
    :param rast_list: list of the raster paths, geomorphic indices and sinkholes mask
    :param output_folder: a folder in which we will save all the outputs
    :param file_name: the name of the output files
    :return:
    """
    # Check if the folder exists
    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")
    else:
        print(f"Folder '{output_folder}' already exists.")
    # transforming rasters to one table with coordinates, features and mask
    table_out = output_folder + '/' + file_name + '_table.npy'
    raster_to_table(rast_list, table_out)
    # filtering the features columns (not the mask and coordinates)
    filter_out = output_folder + '/' + file_name + '_table_filtered.npy'
    filter_table_by_std(table_out, filter_out, 2)
    sink0 = output_folder + '/' + file_name + 'sinkholes0_filtered.npy'
    sink1 = output_folder + '/' + file_name + 'sinkholes1_filtered.npy'
    sink_num = split_binary(filter_out, sink0, sink1)
    sink0_rand = output_folder + '/' + file_name + 'sinkholes0_rand_filtered.npy'
    choose_random(sink0, sink0_rand, sink_num)


def add_rasters_to_table(rast_list, former_table, table_out, output_folder):
    """
    this function takes a rasters list and add the rasters to a table that was formerly created.
    steps:
    takes the rasters list and creating a new table with coordinates for them only.
    filters the raster columns.
    adds the new columns to the former table between the former features and the sinkholes mask.
    :param rast_list:
    :param former_table:
    :param table_out:
    :param output_folder:
    :return:
    """
    # Check if the folder exists
    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")
    else:
        print(f"Folder '{output_folder}' already exists.")
    # creating a new table with coordinates for the rasters
    rast_table = output_folder + '/' + table_out + '_rast_table.npy'
    raster_to_table(rast_list, rast_table)
    print('raster to table finished')
    # adding the new columns to the former table between the former features and the sinkholes mask
    add_columns(former_table, rast_table, rast_table)
    print('adding columns finished')
    # filtering the features columns (not the mask and coordinates)
    filter_out = output_folder + '/' + table_out + '_rast_table_filtered.npy'
    filter_table_by_std(rast_table, filter_out, 2)
    print('filtering finished')
    # divide the table to sinkholes and non-sinkholes
    sink0 = output_folder + '/' + table_out + 'sinkholes0_filtered.npy'
    sink1 = output_folder + '/' + table_out + 'sinkholes1_filtered.npy'
    sink_num = split_binary(table_out, sink0, sink1)
    print('splitting finished')
    # choosing random non-sinkholes
    sink0_rand = output_folder + '/' + table_out + 'sinkholes0_rand_filtered.npy'
    choose_random(sink0, sink0_rand, sink_num)
    print('choosing random finished')
    # deleting the filtered table and the rast table
    os.remove(rast_table)
    os.remove(filter_out)


def add_columns(former_table, new_table, table_out):
    """
    this function takes a table and adds its columns to another table .
    the new columns will be added to the former table between the former features and the sinkholes mask.
    :param former_table: numpy array
    :param new_table: numpy array
    :param table_out: location to save the new table
    :return:
    """
    # loading the tables
    former = np.load(former_table)
    new = np.load(new_table)
    # creating a new table with the same shape as the former table
    new_table = np.zeros([former.shape[0], former.shape[1] + new.shape[1] - 2])
    # adding the former table to the new table
    new_table[:, 0:former.shape[1] - 1] = former[:, 0:former.shape[1] - 1]
    new_table[:, -1] = former[:, -1]
    # adding the new columns to the new table
    new_table[:, former.shape[1] - 1:-1] = new[:, 2:]
    # saving the new table
    np.save(table_out, new_table)


def low_pass_filter(input_raster, output_raster, filter_size):
    """
    This function applies a low pass filter on a raster using convolution.
    :param input_raster: a raster file
    :param output_raster:
    :param filter_size:
    """
    # Read the input raster using gdal
    dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
    image_array = dataset.ReadAsArray()
    proj = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    print(nodata)
    # if there is the value -32767 in the raster, it will be replaced with 0
    image_array[image_array == -32767] = 0
    image_array[image_array == nodata] = 0

    # Create a 2D mean filter kernel
    kernel = np.ones((filter_size, filter_size)) / (filter_size * filter_size)

    # Apply the mean filter using convolution
    filtered_image = ndimage.convolve(image_array, kernel)

    # Save the filtered image to the output raster
    save_array_as_tiff(filtered_image, output_raster, geotransform, proj)


def min_max_normalization(input_path, output_path, minmax=None):
    # Open the input raster using GDAL
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    proj = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    if dataset is None:
        print("Error: Could not open the raster dataset.")
        return

    # Read raster data into a NumPy array
    raster_array = dataset.ReadAsArray()

    # Get the minimum and maximum values in the raster array
    if minmax is None:
        min_value = np.min(raster_array)
        max_value = np.max(raster_array)
    else:
        min_value = minmax[0]
        max_value = minmax[1]

    # Normalize the raster values
    normalized_array = (raster_array - min_value) / (max_value - min_value)

    save_array_as_tiff(normalized_array, output_path, geotransform, proj)

from osgeo import gdal
import numpy as np
import os

def normalize_rasters(input_folder, output_folder):
    # Function to normalize an array
    def normalize_array(array, min_val, max_val):
        return (array - min_val) / (max_val - min_val)

    # List all raster files in the input folder
    raster_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".tif")]

    # Read all rasters and store them in a list
    raster_arrays = []
    for raster_file in raster_files:
        dataset = gdal.Open(raster_file)
        if dataset is None:
            print(f"Failed to open {raster_file}. Skipping.")
            continue
        raster_array = dataset.ReadAsArray()
        # flatten the array
        raster_array = raster_array.flatten()
        raster_arrays.append(raster_array)
        dataset = None

    # Concatenate all arrays into a single array
    concatenated_array = np.concatenate(raster_arrays, axis=0)
    # remove -32767 values
    concatenated_array = concatenated_array[concatenated_array != -32767]
    concatenated_array = concatenated_array[concatenated_array != -99999]

    # Calculate global minimum and maximum values
    global_min = np.min(concatenated_array)
    global_max = np.max(concatenated_array)

    print(f'Global min: {global_min}')
    print(f'Global max: {global_max}')

    # go through the rasters again and normalize them
    for raster_file in raster_files:
        dataset = gdal.Open(raster_file)
        if dataset is None:
            print(f"Failed to open {raster_file}. Skipping.")
            continue
        raster_array = dataset.ReadAsArray()

        normalized_array = normalize_array(raster_array, global_min, global_max)
        # Save the normalized array to a new raster file
        normalized_array[raster_array == -32767] = -32767
        normalized_array[raster_array == -99999] = -32767
        normalized_array[raster_array == -9999] = -32767
        output_file = os.path.join(output_folder, os.path.basename(raster_file))
        save_array_as_tiff(normalized_array, output_file, dataset.GetGeoTransform(), dataset.GetProjection())
        dataset = None


    print("Normalization completed successfully.")


def raster_sens_spec(pred_raster, true_raster):
    # Open the predicted raster using GDAL
    pred_dataset = gdal.Open(pred_raster, gdal.GA_ReadOnly)
    pred_array = pred_dataset.ReadAsArray()
    # Open the true raster using GDAL
    true_dataset = gdal.Open(true_raster, gdal.GA_ReadOnly)
    true_array = true_dataset.ReadAsArray()
    # Calculate the number of true positives, false positives, true negatives, and false negatives
    tp = np.sum(np.logical_and(pred_array == 1, true_array == 1))
    fp = np.sum(np.logical_and(pred_array == 1, true_array == 0))
    tn = np.sum(np.logical_and(pred_array == 0, true_array == 0))
    fn = np.sum(np.logical_and(pred_array == 0, true_array == 1))
    print("True positives: " + str(tp))
    print("False positives: " + str(fp))
    print("True negatives: " + str(tn))
    print("False negatives: " + str(fn))
    # Calculate the sensitivity and specificity
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print("Sensitivity: " + str(sens))
    print("Specificity: " + str(spec))
    return tp, fp, tn, fn, sens, spec


def raster_sens_spec2(pred_raster, true_raster):
    # Open the predicted raster using GDAL
    pred_dataset = gdal.Open(pred_raster, gdal.GA_ReadOnly)
    pred_array = pred_dataset.ReadAsArray()
    # Open the true raster using GDAL
    true_dataset = gdal.Open(true_raster, gdal.GA_ReadOnly)
    true_array = true_dataset.ReadAsArray()

    # Masking values equal to -32767
    valid_mask = (pred_array != -32767) & (true_array != -32767)
    pred_array = pred_array[valid_mask]
    true_array = true_array[valid_mask]

    # Calculate the number of true positives, false positives, true negatives, and false negatives
    tp = np.sum(np.logical_and(pred_array == 1, true_array == 1))
    fp = np.sum(np.logical_and(pred_array == 1, true_array == 0))
    tn = np.sum(np.logical_and(pred_array == 0, true_array == 0))
    fn = np.sum(np.logical_and(pred_array == 0, true_array == 1))

    print("True positives: " + str(tp))
    print("False positives: " + str(fp))
    print("True negatives: " + str(tn))
    print("False negatives: " + str(fn))

    # Calculate the sensitivity and specificity
    sens = tp / (tp + fn) if tp + fn != 0 else 0  # Handle division by zero
    spec = tn / (tn + fp) if tn + fp != 0 else 0  # Handle division by zero

    print("Sensitivity: " + str(sens))
    print("Specificity: " + str(spec))

    return tp, fp, tn, fn, sens, spec

def pred2polygons(pred_raster, output_shapefile):
    ds = gdal.Open(pred_raster)
    if ds is None:
        raise Exception("Failed to open the raster file.")
    band = ds.GetRasterBand(1)
    proj = ds.GetProjection()
    shp_proj = osr.SpatialReference(wkt=proj)
    shp_proj.MorphToESRI()
    driver = ogr.GetDriverByName("ESRI Shapefile")
    create_shp = driver.CreateDataSource(output_shapefile)
    shp_layer = create_shp.CreateLayer(output_shapefile, srs=shp_proj)
    new_field = ogr.FieldDefn("DN", ogr.OFTInteger)
    shp_layer.CreateField(new_field)
    gdal.Polygonize(band, None, shp_layer, 0, [], callback=None)
    create_shp.Destroy()


def del_0val_polygons(shapefile_path):
    """

    :param shapefile_path:
    :return:
    """
    ds = ogr.Open(shapefile_path, 1)  # 1 for read-write mode
    if ds is None:
        raise Exception("Failed to open the shapefile.")
    layer = ds.GetLayer()
    target_values = [0, -32767]  # Values to delete
    to_delete = []
    for feature in layer:
        if feature.GetField("DN") in target_values:
            to_delete.append(feature.GetFID())  # Store feature IDs to delete

    # Delete the selected features
    for fid in to_delete:
        layer.DeleteFeature(fid)
    del ds


def fill_holes(shapefile_path, output_shapefile):
    gdf = gpd.read_file(shapefile_path)

    def remove_holes(polygon):
        if polygon.geom_type == 'Polygon':
            return Polygon(polygon.exterior)
        elif polygon.geom_type == 'MultiPolygon':
            return MultiPolygon([Polygon(poly.exterior) for poly in polygon])
        else:
            return polygon  # Return unchanged for other geometry types

    gdf['geometry'] = gdf['geometry'].apply(remove_holes)
    gdf.to_file(output_shapefile)


def pred_to_raster(input_raster, input_shapefile, output_raster):
    # Open the input raster using GDAL
    raster_dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
    if raster_dataset is None:
        raise Exception("Failed to open the raster dataset.")
    # Get the geotransform and projection
    geotransform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjection()
    # Create a new raster dataset
    driver = gdal.GetDriverByName("GTiff")
    out_raster = driver.Create(output_raster, raster_dataset.RasterXSize, raster_dataset.RasterYSize, 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(proj)
    # Rasterize the shapefile
    gdal.RasterizeLayer(out_raster, [1], ogr.Open(input_shapefile).GetLayer(), burn_values=[1])
    out_raster.FlushCache()
    out_raster = None


def fill_preds(pred_raster, output_raster, output_shp):
    first_shp = output_shp + '_1st.shp'
    pred2polygons(pred_raster, first_shp)
    del_0val_polygons(first_shp)
    fill_holes(first_shp, output_shp)
    rasterize_shapefile(output_shp, pred_raster, output_raster)


def rasterize_shapefile(shapefile_path, raster_template_path, output_raster_path, attribute_name='DN'):
    # Open shapefile
    shape_ds = ogr.Open(shapefile_path)
    if shape_ds is None:
        raise Exception(f"Failed to open shapefile: {shapefile_path}")

    # Open raster template
    raster_template_ds = gdal.Open(raster_template_path)
    if raster_template_ds is None:
        raise Exception(f"Failed to open raster template: {raster_template_path}")

    # Create output raster
    output_ds = gdal.GetDriverByName('GTiff').Create(output_raster_path,
                                                    raster_template_ds.RasterXSize,
                                                    raster_template_ds.RasterYSize,
                                                    1,  # Number of bands
                                                    gdal.GDT_Byte)
    output_ds.SetProjection(raster_template_ds.GetProjection())
    output_ds.SetGeoTransform(raster_template_ds.GetGeoTransform())

    # Create in-memory raster from shapefile
    mem_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    mem_layer = mem_ds.CreateLayer('mem_layer', srs=shape_ds.GetLayer().GetSpatialRef(), geom_type=ogr.wkbPolygon)
    mem_layer.CreateField(ogr.FieldDefn(attribute_name, ogr.OFTInteger))

    for feature in shape_ds.GetLayer():
        mem_layer.CreateFeature(feature)

    # Rasterize shapefile onto the output raster
    # gdal.RasterizeLayer(output_ds, [1], mem_layer, burn_values=[1], options=["ATTRIBUTE=" + attribute_name])
    gdal.RasterizeLayer(output_ds, [1], mem_layer, burn_values=[1])

    # Clean up
    shape_ds = None
    raster_template_ds = None
    mem_ds = None
    output_ds = None




class LRmodel:
    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.coeffs = None
        self.intercept = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coeffs = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.n_iters):
            z = self.intercept + np.dot(X, self.coeffs)  # z = a + bx
            predictions = self.sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.coeffs -= self.lr * dw
            self.intercept -= self.lr * db

    def predict(self, X, threshold=0.5):
        lin_prob = self.intercept + np.dot(X, self.coeffs)
        y_prob = self.sigmoid(lin_prob)
        y_pred = [0 if y <= threshold else 1 for y in y_prob]

        return y_pred

    def predict_proba(self, X):
        lin_prob = self.intercept + np.dot(X, self.coeffs)
        y_prob = self.sigmoid(lin_prob)

        return y_prob


# example of using the LRmodel class
# if __name__ == "__main__":
#     # load the data
#     X, y = load_data('C:/Users/97254/PycharmProjects/untitled1/venv/Scripts/tables/tables/tables/
#     table_2019_2
#     .npy')
#     # split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#     # train the model
#     model = LRmodel(lr=0.0001, n_iters=1000)
#     model.fit(X_train, y_train)
#     # predict the test set
#     y_pred = model.predict(X_test)
#     # calculate the accuracy
#     accuracy = accuracy(y_pred, y_test)
#     print(accuracy)


def accuracy(y_pred, y_test):
    conf_mat = confusion_matrix(y_test, y_pred)
    acc = (conf_mat[0, 0] + conf_mat[1, 1]) / (conf_mat[0, 0] + conf_mat[1, 1] + conf_mat[0, 1] + conf_mat[1, 0])
    sensitivity = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    specificity = conf_mat[0, 0] / (conf_mat[0, 1] + conf_mat[0, 0])
    fpr = conf_mat[0, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    return acc, sensitivity, specificity, fpr, conf_mat

# if __name__ == "__main__":
# input = sys.argv[1]
# radius = sys.argv[2]
# output = sys.argv[3]
# elevation_per(input, radius, output)

# pred_to_raster('/Users/nirgut/Documents/Master_Files/Files/train_test/dem_train.tif','/Users/nirgut/Documents/Master_Files/Files/train_test/probs.npy','/Users/nirgut/Documents/Master_Files/Files/train_test/prob.tif')

# implement logistic regression
# if __name__ == "__main__":
#     # load the data
#     X, y = load_data('C:/Users/97254/PycharmProjects/untitled1/venv/Scripts/tables/tables/tables/table_2019_2.npy')
#     # split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#     # train the model


def extract_polygon(input_shapefile, output_shapefile, polygon_index):
    # Read the input shapefile
    gdf = gpd.read_file(input_shapefile)

    # Extract the desired polygon by index
    selected_polygon = gdf.iloc[polygon_index]['geometry']

    # Create a new GeoDataFrame with the selected polygon
    new_gdf = gpd.GeoDataFrame(geometry=[selected_polygon])

    # Save the new GeoDataFrame to a new shapefile
    new_gdf.to_file(output_shapefile, driver='ESRI Shapefile')


def create_mask_from_shapefile(shapefile_path, raster_path, output_path, nodata_value=-32767):
    """
    Create a mask from a shapefile and raster, saving the result to a new file with a specified nodata value.
    """
    # Open the raster file
    raster = gdal.Open(raster_path)
    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize

    # Create the output raster
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(projection)

    # Create a layer from the shapefile
    shapefile = ogr.Open(shapefile_path)
    layer = shapefile.GetLayer()

    # Rasterize the shapefile layer to the output raster
    gdal.RasterizeLayer(out_raster, [1], layer, burn_values=[1])

    # Read the mask
    band = out_raster.GetRasterBand(1)
    mask = band.ReadAsArray()

    # Apply the nodata value to the mask
    mask[mask == 0] = nodata_value
    band.WriteArray(mask)
    band.SetNoDataValue(nodata_value)

    out_raster.FlushCache()
    out_raster = None
    shapefile = None


def extract_rasters_by_polygons(shapefile_path, raster_path, output_folder, nodata_value=-32767):
    """
    Extract rasters by polygons and save them to an output folder with a specified nodata value.
    """
    # Check if the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the input shapefile
    gdf = gpd.read_file(shapefile_path)

    # Loop through the polygons
    for i in range(len(gdf)):
        # file_path = os.path.join(output_folder, f"{gdf.iloc[i]['number']}.tif")
        file_path = os.path.join(output_folder, f"new_file.tif")
        print(file_path)
        extract_polygon(shapefile_path, 'temp.shp', i)
        create_mask_from_shapefile('temp.shp', raster_path, file_path, nodata_value)

    os.remove('temp.shp')


### contours algorithm functions:

def get_all_contours(level, data, extent=None, plot=False):
    """
    :param level: desired level contour
    :param data: raster values as np.array
    :param extent: coordinates (x0, x1, y0, y1) or (minlon, maxlon, minlat, maxlat)
    :param plot: Boolean. Should we plot?
    :return: List of coordinates, each containing two columns (x, y or lon, lat)
    """
    # Create a figure
    fig, ax = plt.subplots()

    # Generate contours
    cs = ax.contour(data, levels=[level], extent=extent)

    # Store coordinates of all contours
    all_coords = []

    # Iterate through each contour
    for contour in cs.collections:
        for path in contour.get_paths():
            vertices = path.vertices
            x, y = vertices[:, 0], vertices[:, 1]
            coords = np.column_stack((x, y))
            all_coords.append(coords)

    if plot:
        # Plot all contours
        for coords in all_coords:
            ax.plot(coords[:, 0], coords[:, 1])

        plt.show()
    else:
        plt.close(fig)

    return all_coords


def get_contour_values(all_coords, data):
    """
    :param all_coords: List of coordinates for each contour
    :param data: Original matrix
    :return: List of values from the original matrix at the contour coordinates
    """
    contour_values = []

    for coords in all_coords:
        # Convert coordinates to integer indices
        indices = np.round(coords).astype(int)

        # Extract values from the original matrix
        values = data[indices[:, 1], indices[:, 0]]

        contour_values.append(values)

    return contour_values


def classify_contour(contour):
    """
    Classify a contour as a polygon or a line.

    Parameters:
    - contour: NumPy array representing the contour vertices (2D array).

    Returns:
    - 'polygon' if the contour is closed (forms a polygon).
    - 'line' if the contour is open (forms a line).
    - 'unknown' if the classification is ambiguous or fails.
    """
    if len(contour) < 3:
        return 'unknown'  # Contour must have at least 3 points

    # Check if the first and last points are the same (closed contour)
    closed = np.all(contour[0] == contour[-1])

    if closed:
        return True
    else:
        return False


def ras2multiple(raster_path, file_name, tilesize):
    dset = gdal.Open(raster_path)
    print(os.getcwd())
    # change the working directory to the output folder
    os.chdir(os.path.dirname(raster_path))

    width = dset.RasterXSize
    height = dset.RasterYSize
    # calculate the total number of tiles
    cols = width // tilesize
    rows = height // tilesize
    print(f'cols: {cols}, rows: {rows}')

    tile_number = 0
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            tile_number += 1
            w = min(i + tilesize, width) - i
            h = min(j + tilesize, height) - j
            gdaltranString = ["gdal_translate", "-of", "GTIFF", "-srcwin", str(i), str(j), str(w), str(h),
                              raster_path, f"{file_name}_{i}_{j}.tif"]
            subprocess.run(gdaltranString)
            print(f"Tile {i}_{j} created")
            print(f'tile number: {tile_number} out of {cols * rows}')


def rasterize_polygon(input_shapefile, input_raster, output_raster):
    # Open the shapefile
    shape_ds = ogr.Open(input_shapefile)
    if shape_ds is None:
        raise Exception(f"Failed to open shapefile: {input_shapefile}")

    # Open the raster file
    raster_ds = gdal.Open(input_raster, gdalconst.GA_ReadOnly)
    if raster_ds is None:
        raise Exception(f"Failed to open raster file: {input_raster}")

    # Get raster properties
    raster_cols = raster_ds.RasterXSize
    raster_rows = raster_ds.RasterYSize
    raster_proj = raster_ds.GetProjection()
    raster_geo_transform = raster_ds.GetGeoTransform()

    # Create a new raster dataset
    driver = gdal.GetDriverByName('GTiff')
    new_raster_ds = driver.Create(output_raster, raster_cols, raster_rows, 1, gdal.GDT_Int16)
    new_raster_ds.SetProjection(raster_proj)
    new_raster_ds.SetGeoTransform(raster_geo_transform)

    # Set the no data value
    new_raster_ds.GetRasterBand(1).SetNoDataValue(0)

    # Set all values to 0 initially
    new_raster_ds.GetRasterBand(1).WriteArray(np.zeros((raster_rows, raster_cols), dtype=np.int16))

    # Rasterize the shapefile and set the values inside polygons to 1
    gdal.RasterizeLayer(new_raster_ds, [1], shape_ds.GetLayer(), burn_values=[1])

    # Close datasets
    shape_ds = None
    raster_ds = None
    new_raster_ds = None

    print(f"Rasterization completed. Output saved to: {output_raster}")


def model_post_process(rasters_folder, npy_folder, fit_model, output_folder, drop, threshold=0.5):
    """
    this function will get the fitted model alongside the folder of test rasters and will return the predicted rasters
    after apllying the filter_small_obj function and the fill_holes function.
    it will also calculate the sensitivity and specificity for all of the rasters together after the post processing.

    :param rasters_folder: will be used to create the reference frame for the creation of predicted rasters
    :param npy_folder: will be used as the input for the prediction of the model
    :param fit_model: pre trained logistic regression model
    :param drop: the dropped columns for the npy files
    :param output_folder: folder for the outputs of the function
    :return:
    """
    # check if the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # load the model
    # with open(fit_model, 'rb') as file:
    #     model = pickle.load(file)
    # load the rasters
    rasters = os.listdir(rasters_folder)
    # loop through the rasters
    for raster in rasters:
        # load the raster
        raster_path = rasters_folder + '/' + raster
        raster_name = raster.split('.')[0]
        # load the npy file
        npy_path = npy_folder + '/' + raster_name + '.npy'
        npy = np.load(npy_path)
        # drop the columns
        npy = np.delete(npy, drop, axis=1)
        # predict the npy file
        probs = fit_model.predict_proba(npy)
        # save the predicted raster
        pred_raster = output_folder + '/' + raster_name + '_pred.tif'
        # predict using the threshold and probs
        pred_array = np.where(probs[:, 1] >= threshold, 1, 0)
        pred_array2raster(raster_path, pred_array, pred_raster)
        # filter the small objects
        filtered_raster = output_folder + '/' + raster_name + '_filtered.tif'
        filter_small_obj(pred_raster, filtered_raster, 100)
        # fill the holes
        filled_raster = output_folder + '/' + raster_name + '_filled.tif'
        fill_holes(filtered_raster, filled_raster)
        # calculate the sensitivity and specificity
        sens, spec = raster_sens_spec(filled_raster, raster_path)
        print(f'{raster_name} sensitivity: {sens}, specificity: {spec}')


# from pyproj import Proj, transform
# import numpy as np
# import pandas as pd
#
# # Load .npy files
# array1 = np.load('/Users/nirgut/Documents/Master_Files/Raster_Files/clipped_examples/all_aois/train_DSMnorm_rivers.npy')
# array2 = np.load('/Users/nirgut/Documents/Master_Files/Raster_Files/clipped_examples/all_aois/test_DSMnorm_rivers.npy')
# print(array1[0:5,:])
# # convert columns 0 and 1 to wgs84
# def convert_to_wgs84(x, y):
#     in_proj = Proj(init='epsg:2039')
#     out_proj = Proj(init='epsg:4326')  # WGS84
#     print(x, y)
#     print(type(x), type(y))
#     lon, lat = transform(in_proj, out_proj, y, x)
#     return lon, lat
# # use the function
# array1[:, 0], array1[:, 1] = convert_to_wgs84(array1[:, 0], array1[:, 1])
# array2[:, 0], array2[:, 1] = convert_to_wgs84(array2[:, 0], array2[:, 1])
#
# # Drop rows with -32767
# array1 = array1[array1[:, 2] != -32767]
#
# # Combine arrays vertically
# combined_array = np.vstack((array1, array2))
#
# # Add an ID column
# id_column = np.arange(1, combined_array.shape[0]+1).reshape(-1, 1)
# combined_array_with_id = np.hstack((id_column, combined_array))
#
#
#
# # Convert to pandas DataFrame
# df = pd.DataFrame(combined_array_with_id)
#
# # print 5 first rows
# print(df.head())
# out = '/Users/nirgut/Documents/Master_Files/Raster_Files/clipped_examples/all_aois/train_test_combined_ver2.csv'

##################### MORPHOMETRIC FEATURES ALGORITHMS ############################
def contour_circularity(in_raster_path, out_path, min_coord, min_area=0):
    """
    This function calculates the circularity of the contours in the raster.
    it goes through each pixel and calculates the circularity of the contours that contain the pixel.

    :param in_raster_path:
    :param out_path:
    :param min_coord:
    :return:
    """
    raster = gdal.Open(in_raster_path)
    width = raster.RasterXSize
    height = raster.RasterYSize
    print(f"Width: {width}, Height: {height}")
    projection = raster.GetProjection()
    geotransform = raster.GetGeoTransform()
    array = raster.ReadAsArray()

    out_mat = np.full_like(array, -5)
    # print(out_mat[0,177])

    for i in range(array.shape[0]):
        # print only if the row is a multiple of 50
        if i % 50 == 0:
            print(f'row: {i}')
        for j in range(array.shape[1]):
            if out_mat[i, j] == -5 and array[i, j] != -32767:
              # print(f'i: {i}, j: {j}')
              # print(f'value is: {array[i, j]}')
                temp_contours = get_all_contours(array[i, j], array)
                for contour in temp_contours:
                    is_closed = classify_contour(contour)
                    round_coords = np.floor(contour).astype(int)
                    if is_closed:
                        if len(contour) < min_coord:
                            for coord in round_coords:
                                if out_mat[coord[1], coord[0]] == -5:
                                    out_mat[coord[1], coord[0]] = 0
                        else:

                            poly = Polygon(contour)
                            area = poly.area
                            if area < min_area:
                                for coord in round_coords:
                                    out_mat[coord[1], coord[0]] = 0
                            else:
                                perimeter = poly.length
                                if perimeter != 0:
                                    circularity = (4 * np.pi * area)/(perimeter**2)
                                    for coord in round_coords:
                                        out_mat[coord[1], coord[0]] = circularity

                    elif not is_closed:
                        for coord in round_coords:
                            out_mat[coord[1], coord[0]] = 0
                    # if 0 <= coord[0] < out_mat.shape[0] and 0 <= coord[1] < out_mat.shape[1]:
                    #   out_mat[coord[1], coord[0]] = 0
                    # else:
                    #   out_mat[coord[1], coord[0]] = 0.1
                    else:
                        for coord in round_coords:
                            out_mat[coord[1], coord[0]] = 0
            elif out_mat[i, j] == -5 and array[i, j] == -32767:
                out_mat[i, j] = -32767
              # print(out_mat[i, j])
    save_array_as_tiff(out_mat, out_path, geotransform, projection, nodata_val=-32767)


def normalize_array(array):
    """Normalize the array values to the range [0, 1]."""
    array_min = array.min()
    array_max = array.max()
    if array_max - array_min == 0:
        return array  # Avoid division by zero if all values are the same
    normalized_array = (array - array_min) / (array_max - array_min)
    return normalized_array


def divide_raster_to_tiles(input_raster_path, output_dir, tile_size_x, tile_size_y, overlap_x=0, overlap_y=0):
    # Open the input raster
    raster = gdal.Open(input_raster_path)
    if raster is None:
        print(f"Unable to open input raster {input_raster_path}")
        return

    # Get raster information
    raster_band = raster.GetRasterBand(1)
    xsize = raster.RasterXSize
    ysize = raster.RasterYSize
    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()

    # Calculate the number of tiles in x and y directions, considering overlap
    x_tiles = (xsize + tile_size_x - 1) // (tile_size_x - overlap_x)
    y_tiles = (ysize + tile_size_y - 1) // (tile_size_y - overlap_y)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in tqdm.tqdm(range(x_tiles)):
        for j in range(y_tiles):
            # Calculate the offsets and tile size
            x_offset = i * (tile_size_x - overlap_x)
            y_offset = j * (tile_size_y - overlap_y)
            x_tile_size = min(tile_size_x, xsize - x_offset)
            y_tile_size = min(tile_size_y, ysize - y_offset)

            # Ensure that the tile dimensions are positive
            if x_tile_size <= 0 or y_tile_size <= 0:
                continue

            # Read the tile data
            tile = raster_band.ReadAsArray(x_offset, y_offset, x_tile_size, y_tile_size)

            # Normalize the tile data
            # tile = normalize_array(tile)

            # Create a new raster dataset for the tile
            driver = gdal.GetDriverByName('GTiff')
            output_tile_path = os.path.join(output_dir, f'tile_{i}_{j}.tif')
            tile_dataset = driver.Create(output_tile_path, x_tile_size, y_tile_size, 1, gdal.GDT_Float32)
            tile_dataset.SetGeoTransform((
                geotransform[0] + x_offset * geotransform[1],
                geotransform[1],
                0,
                geotransform[3] + y_offset * geotransform[5],
                0,
                geotransform[5]
            ))
            tile_dataset.SetProjection(projection)
            tile_dataset.GetRasterBand(1).WriteArray(tile)
            # set the nodata value
            tile_dataset.GetRasterBand(1).SetNoDataValue(-32767)

            # Save and close the tile dataset
            tile_dataset.FlushCache()
            tile_dataset = None

    # Close the original raster
    raster = None


def check_empty_rasters(raster_dir, threshold, DSM_dir=None):
    """
    Check if rasters in a directory are empty or not based on a threshold and organize them accordingly.

    :param raster_dir: Directory containing the raster files.
    :param threshold: Threshold value to determine if a raster is considered empty.
    :param DSM_dir: Directory containing the corresponding DSM files (optional).
    """
    # Create the empty and not_empty directories for both raster and DSM directories
    empty_dir = os.path.join(raster_dir, 'empty')
    not_empty_dir = os.path.join(raster_dir, 'not_empty')
    nodata_dir = os.path.join(raster_dir, 'nodata')

    if DSM_dir:
        empty_DSM_dir = os.path.join(DSM_dir, 'empty')
        not_empty_DSM_dir = os.path.join(DSM_dir, 'not_empty')
        nodata_DSM_dir = os.path.join(DSM_dir, 'nodata')

    for dir_path in [empty_dir, not_empty_dir, nodata_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if DSM_dir:
        for dir_path in [empty_DSM_dir, not_empty_DSM_dir, nodata_DSM_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    print(f'The size of the directory is: {len(os.listdir(raster_dir))}')

    # Iterate over the rasters in the directory
    for raster in tqdm.tqdm(os.listdir(raster_dir)):
        if raster.endswith('.tif'):
            raster_path = os.path.join(raster_dir, raster)
            # Open the raster
            dataset = gdal.Open(raster_path)
            # Get the band
            band = dataset.GetRasterBand(1)
            # Read the array
            array = band.ReadAsArray()
            # Count the number of non-zero pixels
            non_zero_count = (array != 0).sum()
            nodata_count = (array == -32767).sum()
            # print(f'Raster: {raster}, Non-zero count: {non_zero_count}')
            # Determine the destination based on the non-zero count
            if non_zero_count < threshold:
                os.rename(raster_path, os.path.join(empty_dir, raster))
                # print('empty')
                if DSM_dir:
                    dsm_path = os.path.join(DSM_dir, raster)
                    if os.path.exists(dsm_path):
                        os.rename(dsm_path, os.path.join(empty_DSM_dir, raster))
            else:
                if nodata_count > 0:
                    os.rename(raster_path, os.path.join(nodata_dir, raster))
                    if DSM_dir:
                        dsm_path = os.path.join(DSM_dir, raster)
                        if os.path.exists(dsm_path):
                            os.rename(dsm_path, os.path.join(nodata_DSM_dir, raster))
                else:
                    os.rename(raster_path, os.path.join(not_empty_dir, raster))

                    if DSM_dir:
                        dsm_path = os.path.join(DSM_dir, raster)
                        if os.path.exists(dsm_path):
                            os.rename(dsm_path, os.path.join(not_empty_DSM_dir, raster))



def split_data_and_masks(data_folder, mask_folder, train_ratio=0.8):
    """
    Split data and mask files into training and testing sets.
    :param data_folder:
    :param mask_folder:
    :param train_ratio:
    :return:
    """
    # Create train and test directories
    train_data_dir = os.path.join(os.path.dirname(data_folder), 'train_data')
    test_data_dir = os.path.join(os.path.dirname(data_folder), 'test_data')
    train_mask_dir = os.path.join(os.path.dirname(mask_folder), 'train_masks')
    test_mask_dir = os.path.join(os.path.dirname(mask_folder), 'test_masks')

    for dir_path in [train_data_dir, test_data_dir, train_mask_dir, test_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of files
    data_files = os.listdir(data_folder)
    mask_files = os.listdir(mask_folder)
    # Determine number of files for training
    num_train = int(len(data_files) * train_ratio)

    # Shuffle indices
    indices = list(range(len(data_files)))
    random.shuffle(indices)

    # Split data
    train_data_files = [data_files[i] for i in indices[:num_train]]
    test_data_files = [data_files[i] for i in indices[num_train:]]

    # Copy data files to train and test directories
    for file_name in train_data_files:
        shutil.copy(os.path.join(data_folder, file_name), os.path.join(train_data_dir, file_name))
    for file_name in test_data_files:
        shutil.copy(os.path.join(data_folder, file_name), os.path.join(test_data_dir, file_name))

    # Copy mask files to train and test directories
    for file_name in train_data_files:
        if file_name.endswith('.tif'):
            mask_file = file_name.split('.')[0] + '.tif'  # Adjust this based on your file extension
            shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(train_mask_dir, mask_file))

    for file_name in test_data_files:
        if file_name.endswith('.tif'):
            mask_file = file_name.split('.')[0] + '.tif'  # Adjust this based on your file extension
            shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(test_mask_dir, mask_file))

    print("Data split successfully.")


##################### MORPHOMETRIC FEATURES ALGORITHMS ############################
def count_ones(raster_path):
    """
    Count the number of ones in a binary raster.
    :param raster_path:
    :return:
    """
    dataset = gdal.Open(raster_path)
    band = dataset.GetRasterBand(1)
    ones_count = band.ReadAsArray().sum()
    return ones_count


def normalize_raster(input_raster_path, output_raster_path):
    """
    Normalize the values of a raster to the range 0-1 and save it as a new raster.

    :param input_raster_path: Path to the input raster file.
    :param output_raster_path: Path to the output raster file.
    """
    # Open the input raster
    dataset = gdal.Open(input_raster_path, gdalconst.GA_ReadOnly)
    if not dataset:
        print(f"Failed to open raster file: {input_raster_path}")
        return

    # Read the raster data
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    # Normalize the array to the range 0-1
    min_val = np.min(array)
    max_val = np.max(array)
    norm_array = (array - min_val) / (max_val - min_val)

    # Create the output raster
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_raster_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
    if not out_dataset:
        print(f"Failed to create output raster file: {output_raster_path}")
        return

    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())

    # Write the normalized data to the output raster
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(norm_array)
    out_band.SetNoDataValue(-32767)

    # Clean up
    band = None
    dataset = None
    out_band = None
    out_dataset = None


def normalize_rasters_in_folder(input_folder, output_folder):
    """
    Normalize the values of all rasters in a folder to the range 0-1 and save them as new rasters.

    :param input_folder: Path to the folder containing input rasters.
    :param output_folder: Path to the folder where output rasters will be saved.
    """
    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each raster file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            input_raster_path = os.path.join(input_folder, filename)
            output_raster_path = os.path.join(output_folder, filename)
            # print(f"Normalizing {input_raster_path} and saving to {output_raster_path}")
            normalize_raster(input_raster_path, output_raster_path)


