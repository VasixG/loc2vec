import math
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr, osr
from PIL import Image
import json


def meters_to_degrees(lat, meters_lat, meters_lon):
    lat_degrees = meters_lat / 111000
    lon_degrees = meters_lon / (111320 * math.cos(math.radians(lat)))
    return lat_degrees, lon_degrees


def plot_raster_from_tensor(raster_tensor, x_min, x_max, y_min, y_max):
    plt.figure(figsize=(10, 8))
    plt.imshow(raster_tensor, extent=[x_min, x_max, y_min, y_max], cmap='gray')
    plt.title("Rasterized Geometries (Resized to 255x255)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Pixel Value")
    plt.show()


def rasterize_to_tensor(conn, query, pixel_size=0.00007, NoData_value=-99999, target_size=(255, 255)):
    cursor = conn.cursor()
    cursor.execute(query)
    gis_data = cursor.fetchall()
    cursor.close()

    driver = ogr.GetDriverByName("Memory")
    data_source = driver.CreateDataSource("temp_ds")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    layer = data_source.CreateLayer("temp_layer", srs, ogr.wkbUnknown)
    layer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))

    for i, geojson_str in enumerate(gis_data):
        geojson_obj = json.loads(geojson_str[0])
        geometry = ogr.CreateGeometryFromJson(json.dumps(geojson_obj))
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(geometry)
        feature.SetField("ID", i)
        layer.CreateFeature(feature)

    x_min, x_max, y_min, y_max = layer.GetExtent()

    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    band.Fill(NoData_value)

    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[1])

    raster_tensor = band.ReadAsArray()

    resized_image = Image.fromarray(raster_tensor).resize(target_size, Image.Resampling.LANCZOS)
    resized_tensor = np.array(resized_image)

    return resized_tensor


def generate_channel_tensor_and_plot(conn, lat, lon, meters_lon, meters_lat, pixel_size, target_size=(255, 255)):
    lat_deg, lon_deg = meters_to_degrees(lat, meters_lat, meters_lon)

    queries = {
        "roads": f"""
            SELECT AsGeoJSON(
                ST_Intersection(
                    geometry,
                    BuildMbr({lon} - {lon_deg}, {lat} - {lat_deg}, {lon} + {lon_deg}, {lat} + {lat_deg}, 4326)
                )
            )
            FROM lines
            WHERE ST_Intersects(
                geometry,
                BuildMbr({lon} - {lon_deg}, {lat} - {lat_deg}, {lon} + {lon_deg}, {lat} + {lat_deg}, 4326)
            )
            AND highway IS NOT NULL;
        """
    }

    tensor_list = []

    for query in queries.values():
        tensor = rasterize_to_tensor(conn, query, pixel_size, target_size=target_size)
        tensor_list.append(tensor)

    stacked_tensor = np.stack(tensor_list)
    return stacked_tensor
