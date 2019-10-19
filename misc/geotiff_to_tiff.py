from osgeo import gdal, osr
import os
import io
import numpy as np

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,format='GTiff'):

    bands = array.shape[0]
    rows = array.shape[1]
    cols = array.shape[2]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]
    driver = gdal.GetDriverByName(format)
    #Here is where I assign three bands to the raster with int 3
    options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
    outRaster = driver.Create(newRasterfn, cols, rows, bands, gdal.GDT_UInt16, options=options)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    #outband = outRaster.GetRasterBand(1)
    #outband.WriteArray(array)
    for band in range(bands):
        outRaster.GetRasterBand(band+1).WriteArray( array[band, :, :] )

    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(6962)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())

    return outRaster


def example_rgb_creation():
    source = gdal.Open("RGB-PanSharpen_AOI_5_Khartoum_img13.tif")
    rgb_array = source.ReadAsArray()

    return rgb_array

exampleRGB = example_rgb_creation()
exampleOutput = "ExampleOutput.tif"
rasterOrigin=[0,0]
outRaster = array2raster(exampleOutput,rasterOrigin,1, 1,exampleRGB)

outRaster.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
outRaster.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
outRaster.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)


del outRaster
