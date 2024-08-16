import numpy as np
from osgeo import gdal
import time
import math
import sys
import os







A = lambda z1, z2, z3, z4, z5, z6, z7, z8, z9, L: ((z1 + z3 + z7 + z9) / 4 - (z2 + z4 + z6 + z8) / 2 + z5) / (L ** 4)
B = lambda z1, z2, z3, z7, z8, z9, L: ((z1 + z3 - z7 + z9) / 4 - (z2 - z8) / 2) / (L ** 3)
C = lambda z1, z3, z4, z6, z7, z9, L: ((-z1 + z3 - z7 + z9) / 4 + (z4 - z6) / 2) / (L ** 3)
D = lambda z4, z5, z6, L: ((z4 + z6) / 2 - z5) / (L ** 2)
E = lambda z2, z5, z8, L: ((z2 + z8) / 2 - z5) / (L ** 2)
F = lambda z1, z3, z7, z9, L: (-z1 + z3 + z7 - z9) / 4 * (L ** 2)
G = lambda z4, z6, L: (-z4 + z6) / (2 * L)  # in book- Zx
H = lambda z2, z8, L: (z2 - z8) / (2 * L)  # in book- Zy

z_xx = lambda z4, z5, z6, L: (z6 - 2 * z5 + z4) / (L ** 2)
z_xy = lambda z1, z3, z7, z9, L: (-z1 + z3 + z7 - z9) / (4 * L ** 2)
z_yy = lambda z2, z5, z8, L: (z2 - 2 * z5 + z8) / (L ** 2)


def slope_calc(raster):
    rast_src = gdal.Open(raster)
    # rast_src = LiT.reproject_dataset(rast_src)
    # rast_src.FlushCache()
    hill_array = rast_src.ReadAsArray()
    width = range(hill_array.shape[0])
    length = range(hill_array.shape[1])
    slope = np.zeros([hill_array.shape[0], hill_array.shape[1]])
    for row in width[1:-1]:
        for col in length[1:-1]:
            z1 = hill_array[row - 1, col - 1]
            z2 = hill_array[row - 1, col]
            z3 = hill_array[row - 1, col + 1]
            z4 = hill_array[row, col - 1]
            z6 = hill_array[row, col + 1]
            z7 = hill_array[row + 1, col - 1]
            z8 = hill_array[row + 1, col]
            z9 = hill_array[row + 1, col + 1]
            z = np.array([z1, z2, z3, z4, z6, z7, z8, z9])
            z_neigh = z == 0
            L = 0.5  # pixel size in meters
            if not z_neigh.any():
                slope[row, col] = np.degrees(
                    math.atan((math.sqrt(((-z4 + z6) ** 2 + (z2 - z8) ** 2) / (4 * (L ** 2))))))
            else:
                slope[row, col] = 0

    return slope


def aspect(raster):
    rast_src = gdal.Open(raster)
    # rast_src = LiT.reproject_dataset(rast_src)
    # rast_src.FlushCache()
    hill_array = rast_src.ReadAsArray()
    width = range(hill_array.shape[0])
    length = range(hill_array.shape[1])
    aspect_array = np.zeros([hill_array.shape[0], hill_array.shape[1]])
    for row in width[1:-1]:
        for col in length[1:-1]:
            z1 = hill_array[row - 1, col - 1]
            z2 = hill_array[row - 1, col]
            z3 = hill_array[row - 1, col + 1]
            z4 = hill_array[row, col - 1]
            z6 = hill_array[row, col + 1]
            z7 = hill_array[row + 1, col - 1]
            z8 = hill_array[row + 1, col]
            z9 = hill_array[row + 1, col + 1]
            z = np.array([z1, z2, z3, z4, z6, z7, z8, z9])
            z_neigh = z == 0
            if not z_neigh.any():
                asp_val = np.degrees(math.atan2((-z2 + z8), (z4 - z6 + 0.000001)))
                if asp_val < 0:
                    asp_val = 90.0 - asp_val
                elif asp_val > 90.0:
                    asp_val = 360.0 - asp_val + 90.0
                else:
                    asp_val = 90.0 - asp_val
                aspect_array[row, col] = asp_val

            else:
                aspect_array[row, col] = 0
    return aspect_array


def plc(raster, output):
    rast_src = gdal.Open(raster)
    geotrans = rast_src.GetGeoTransform()
    proj = rast_src.GetProjection()
    hill_array = rast_src.ReadAsArray()
    width = range(hill_array.shape[0])
    length = range(hill_array.shape[1])
    print(f'width: {len(width)}')
    print(f'lenght: {len(length)}')
    PLC = np.zeros([hill_array.shape[0], hill_array.shape[1]])
    values_list = np.linspace(1, 200000, num=1000)
    for row in width[1:-1]:
        if row in values_list:
            print(row)
        for col in length[1:-1]:
            # pix_start = time.time()
            z1 = hill_array[row - 1, col - 1]
            z2 = hill_array[row - 1, col]
            z3 = hill_array[row - 1, col + 1]
            z4 = hill_array[row, col - 1]
            z5 = hill_array[row, col]
            z6 = hill_array[row, col + 1]
            z7 = hill_array[row + 1, col - 1]
            z8 = hill_array[row + 1, col]
            z9 = hill_array[row + 1, col + 1]
            z = np.array([z1, z2, z3, z4, z6, z7, z8, z9])
            z_neigh = z == 0
            L = 0.5  # pixel size in meters
            if not z_neigh.any():
                PLC[row, col] = 2 * (
                            D(z4, z5, 6, L) * ((H(z2, z8, L)) ** 2) + E(z2, z5, z8, L) * ((G(z4, z6, L)) ** 2) -
                            F(z1, z3, z7, z9, L) * G(z4, z6, L) * H(z2, z8, L)) / ((G(z4, z6, L)) ** 2 +
                                                                                   (H(z2, z8, L)) ** 2)

            elif z5 is None or z5 == -32767:
                PLC[row, col] = -32767
            else:
                PLC[row, col] = 0
            # pix_end = time.time()
            # print('pixel time: ' + str(pix_end - pix_start))
    print('finished loop')
    # save as tiff
    save_array_as_tiff(PLC, output, geotrans, proj)
    print('finished saving')


def prc(raster, output):
    rast_src = gdal.Open(raster)
    geotrans = rast_src.GetGeoTransform()
    proj = rast_src.GetProjection()
    hill_array = rast_src.ReadAsArray()
    width = range(hill_array.shape[0])
    length = range(hill_array.shape[1])
    PRC = np.zeros([hill_array.shape[0], hill_array.shape[1]])
    start = time.time()
    for row in width[1:-1]:
        if row in [1, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 39999]:
            print(row)
            print(time.time() - start)
        for col in length[1:-1]:
            z1 = hill_array[row - 1, col - 1]
            z2 = hill_array[row - 1, col]
            z3 = hill_array[row - 1, col + 1]
            z4 = hill_array[row, col - 1]
            z5 = hill_array[row, col]
            z6 = hill_array[row, col + 1]
            z7 = hill_array[row + 1, col - 1]
            z8 = hill_array[row + 1, col]
            z9 = hill_array[row + 1, col + 1]
            z = np.array([z1, z2, z3, z4, z6, z7, z8, z9])
            z_neigh = z == 0
            L = 0.5  # pixel size in meters
            if not z_neigh.any():
                PRC[row, col] = 2 * (D(z4, z5, z6, L) * G(z4, z6, L) ** 2 + E(z2, z5, z8, L) * H(z2, z8, L) ** 2
                                     + F(z1, z3, z7, z9, L) * G(z4, z6, L) * H(z2, z8, L)) / (
                                            G(z4, z6, L) ** 2 + H(z2, z8, L) ** 2)
            elif z5 is None or z5 == -32767:
                PRC[row, col] = -32767
            else:
                PRC[row, col] = -32767
    print('finished loop')
    # save as tiff
    save_array_as_tiff(PRC, output, geotrans, proj)
    print('finished saving')
