import torch
from torch.utils.data import DataLoader
from NN import EmbeddingNet, TripletNet, TripletLoss, TripletDataset
from sklearn.model_selection import train_test_split
import random
import numpy as np
import yaml
from fit import fit
from geopy.distance import distance

import os
import re


def get_lat_lon_from_directories(base_dir):
    """
    Extract [lat, lon] from directory names of the format 'loc_{lat}_{lon}'.

    :param base_dir: The base directory where the subdirectories are located
    :return: A list of [lat, lon] coordinates
    """
    lat_lon_list = []

    pattern = r'loc_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)'

    for dir_name in os.listdir(base_dir):
        match = re.match(pattern, dir_name)
        if match:
            lat = float(match.group('lat'))
            lon = float(match.group('lon'))
            lat_lon_list.append([lat, lon])

    return lat_lon_list


def get_near_point(coord, coords, radius_meters):
    """
    Get a random point from the neighborhood within the specified radius.

    :param coord: Tuple (lat, lon) - The center point
    :param coords: Array of tuples [(lat, lon), ...] - Available points
    :param radius_meters: Radius of the neighborhood in meters
    :return: A random nearby point within the specified radius
    """
    neighbors = []

    for point in coords:
        dist = distance(coord, point).meters
        if dist <= radius_meters:
            neighbors.append(point)

    if neighbors:
        return random.choice(neighbors)
    else:
        return None


def get_faraway_point(coord, coords, radius_meters):
    """
    Get a random point from outside the inverse neighborhood (far away from the specified radius).

    :param coord: Tuple (lat, lon) - The center point
    :param coords: Array of tuples [(lat, lon), ...] - Available points
    :param radius_meters: Radius of the inverse neighborhood in meters
    :return: A random point from outside the inverse neighborhood
    """
    far_candidates = []

    for point in coords:
        dist = distance(coord, point).meters
        if dist >= radius_meters:
            far_candidates.append(point)

    if far_candidates:
        return random.choice(far_candidates)
    else:
        return None


def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    coord_with_data = get_lat_lon_from_directories(config.data_dir)

    triplets = []

    for sample in coord_with_data:

        if (near_point := get_near_point(sample, coord_with_data, config['radius_near'])) is None:
            continue

        if (faraway_point := get_faraway_point(sample, coord_with_data, config['radius_far'])) is None:
            continue

        triplets.append([sample, near_point, faraway_point])

    train_features, test_features = train_test_split(
        triplets, test_size=config['test_size'],
        random_state=config['random_seed'])

    triplet_train_dataset = TripletDataset(train_features, config['n_channels'])
    triplet_test_dataset = TripletDataset(test_features, config['n_channels'])

    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=config['batch_size'], shuffle=True, **kwargs)
    triplet_test_loader = DataLoader(triplet_test_dataset, batch_size=config['batch_size'], shuffle=False, **kwargs)

    embedding_net = EmbeddingNet(n_channels=config['n_channels']*3)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()

    loss_fn = TripletLoss(config['margin'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config['scheduler_step'],
        gamma=config['scheduler_gamma'],
        last_epoch=-1)

    train_losses, val_losses = fit(triplet_train_loader, triplet_test_loader, model,
                                   loss_fn, optimizer, scheduler, config['n_epochs'], cuda, config['log_interval'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train Triplet Network on OSM Data")
    parser.add_argument('--config', required=True, help="Path to the config YAML file")
    args = parser.parse_args()

    main(args.config)
