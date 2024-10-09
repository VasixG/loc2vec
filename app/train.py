import torch
from torch.utils.data import DataLoader
from app.nn import EmbeddingNet, TripletNet, TripletLoss
from app.geo_analysis import generate_channel_tensor_and_plot
from app.database import connect_to_spatialite
from sklearn.model_selection import train_test_split
import random
import numpy as np


def generate_mesh(lat_min, lat_max, lon_min, lon_max, num_points_per_axis):
    latitudes = np.linspace(lat_min, lat_max, num_points_per_axis)
    longitudes = np.linspace(lon_min, lon_max, num_points_per_axis)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    return np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T, latitudes, longitudes


def sample_points(mesh, num_samples):
    return random.sample(list(mesh), num_samples)


def get_near_point(lat_idx, lon_idx, latitudes, longitudes):
    neighbors = []

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            if abs(i - lat_idx) <= 2 or abs(j - lon_idx) <= 2:
                neighbors.append(np.array([lat, lon]))

    return random.choice(neighbors)


def get_faraway_point(lat_idx, lon_idx, latitudes, longitudes, min_dist_idx):
    far_candidates = []

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            if abs(i - lat_idx) >= min_dist_idx or abs(j - lon_idx) >= min_dist_idx:
                far_candidates.append(np.array([lat, lon]))

    return random.choice(far_candidates)


def main():
    conn = connect_to_spatialite("/path/to/geodata.db")

    lat_min = 54.65
    lat_max = 54.75
    lon_min = 20.5
    lon_max = 20.55
    num_points_per_axis = 200
    min_dist_idx = 50

    mesh, latitudes, longitudes = generate_mesh(lat_min, lat_max, lon_min, lon_max, num_points_per_axis)

    num_samples = 5
    sampled_points = sample_points(mesh, num_samples)

    triplets = []

    for sample in sampled_points:
        lat_idx = np.where(latitudes == sample[0])[0][0]
        lon_idx = np.where(longitudes == sample[1])[0][0]

    near_point = get_near_point(lat_idx, lon_idx, latitudes, longitudes)

    faraway_point = get_faraway_point(lat_idx, lon_idx, latitudes, longitudes, min_dist_idx)

    triplets.append([sample, near_point, faraway_point])

    train_features, test_features = train_test_split(triplets, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_features, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_features, batch_size=32, shuffle=False)

    model = TripletNet(EmbeddingNet(n_channels=6))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TripletLoss(margin=0.5)

    for epoch in range(10):
        model.train()
        for anchor, positive, negative in train_loader:
            optimizer.zero_grad()
            anchor_out, pos_out, neg_out = model(anchor, positive, negative)
            loss = loss_fn(anchor_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Training loss: {loss.item()}")


if __name__ == '__main__':
    main()
