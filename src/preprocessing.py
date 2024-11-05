import os
import yaml
import logging
import geopandas as gpd
import matplotlib.pyplot as plt
from srai.loaders import OSMOnlineLoader
from srai.regionalizers import geocode_to_region_gdf
from shapely.geometry import box
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_file):
    """
    Load configuration from a YAML file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_poi(tags, location):
    """
    Load POIs from OSM with specified tags.

    :param tags: Dictionary with tags for filtering POIs
    :param location: Location for extracting data from OSM
    :return: Dictionary of GeoDataFrames for each tag
    """
    logging.info("Starting to load POIs from OSM...")
    area = geocode_to_region_gdf(location)
    loader = OSMOnlineLoader()
    gdfs = {}

    tag_pairs = []
    for key, values in tags.items():
        for value in values:
            tag_pairs.append({key: value})

    for tags in tag_pairs:
        key = list(tags.keys())[0]
        value = tags[key]
        try:
            logging.info(f"Loading data for tag {key}={value}")
            gdf = loader.load(area, tags)
            if not gdf.empty:
                tag_name = f"{key}_{value}"
                gdfs[tag_name] = gdf
            else:
                logging.info(f"No data found for tag {key}={value}")
        except Exception as e:
            logging.error(f"Error loading tag {key}={value}: {e}")

    logging.info("Finished loading POIs from OSM.")
    return gdfs, area


def generate_mesh(area_gdf, grid_spacing=1000):
    """
    Generate a mesh grid over the specified area.

    :param area_gdf: GeoDataFrame of the area
    :param grid_spacing: Spacing between grid points in meters
    :return: GeoDataFrame of grid squares and centers in both projected and lat/lon CRS
    """
    area_proj = area_gdf.to_crs(epsg=32637)
    minx, miny, maxx, maxy = area_proj.total_bounds
    minx, miny, maxx, maxy = 414000, 6150000, 427000, 6200000

    x_coords = np.arange(minx, maxx, grid_spacing)
    y_coords = np.arange(miny, maxy, grid_spacing)
    xx, yy = np.meshgrid(x_coords, y_coords)
    x_flat = xx.flatten()
    y_flat = yy.flatten()

    grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_flat, y_flat), crs=area_proj.crs)
    grid_squares = grid_points.copy()
    square_size = 300  # in meters
    grid_squares['geometry'] = grid_squares.apply(
        lambda row: box(
            row.geometry.x - square_size / 2,
            row.geometry.y - square_size / 2,
            row.geometry.x + square_size / 2,
            row.geometry.y + square_size / 2
        ), axis=1
    )

    grid_points_latlon = grid_points.to_crs(epsg=4326)
    return grid_squares, grid_points_latlon


def plot_and_save_images(gdfs, grid_squares, grid_points_latlon, tag_groups):
    """
    Plot and save images for each grid square and tag group.

    :param gdfs: Dictionary of GeoDataFrames for each tag
    :param grid_squares: GeoDataFrame of grid squares
    :param grid_points_latlon: GeoDataFrame of grid points in lat/lon
    :param tag_groups: List of tag groups to plot
    """
    coords_with_data = []
    dir_name = f"data"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logging.info(f"Created directory {dir_name}")
    for idx, square in grid_squares.iterrows():
        center_point = grid_points_latlon.loc[idx].geometry
        lat = round(center_point.y, 6)
        lon = round(center_point.x, 6)

        for i, tag_group in enumerate(tag_groups):
            fig, ax = plt.subplots(figsize=(5, 5))
            minx, miny, maxx, maxy = square.geometry.bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            data_plotted = False

            for tag_name in tag_group:
                if tag_name in gdfs:
                    gdf_tag = gdfs[tag_name].to_crs(grid_squares.crs)
                    gdf_clipped = gpd.clip(gdf_tag, square.geometry)

                    if not gdf_clipped.empty:
                        data_plotted = True
                        gdf_clipped.plot(ax=ax, markersize=5)

            if data_plotted:
                dir_name = f"data/loc_{lat}_{lon}"
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                    # logging.info(f"Created directory {dir_name}")
                coords_with_data.append([lat, lon])

                ax.axis('off')
                image_name = f"img_{lat}_{lon}_{i}.png"
                plt.savefig(os.path.join(dir_name, image_name), bbox_inches='tight', pad_inches=0)
                plt.close()
                # logging.info(f"Saved image {image_name} in {dir_name}")
            else:
                plt.close()
                # logging.info(f"No data for tag group {i} in grid cell centered at ({lat}, {lon})")
    np.save(file='coords_with_data.npy', arr=np.array(coords_with_data))


def main(config_file):
    config = load_config(config_file)
    poi_tags = config['poi_tags']
    location = config['location']
    tag_groups = config['tag_groups']

    gdfs, area = load_poi(poi_tags, location)
    grid_squares, grid_points_latlon = generate_mesh(area, grid_spacing=config['grid_spacing'])
    plot_and_save_images(gdfs, grid_squares, grid_points_latlon, tag_groups)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate OSM-based images for location grid")
    parser.add_argument('--config', required=True, help="Path to the config YAML file")
    args = parser.parse_args()

    main(args.config)
