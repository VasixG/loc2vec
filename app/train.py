import torch
from torch.utils.data import DataLoader
from app.nn import EmbeddingNet, TripletNet, TripletLoss
from app.geo_analysis import generate_channel_tensor_and_plot
from app.database import connect_to_spatialite
from sklearn.model_selection import train_test_split


def main():
    # Load data and model
    conn = connect_to_spatialite("/path/to/geodata.db")

    # Prepare data
    triplets = []  # Add code to generate triplets from mesh grid
    train_features, test_features = train_test_split(triplets, test_size=0.2, random_state=42)

    # Prepare dataset and loaders
    train_loader = DataLoader(train_features, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_features, batch_size=32, shuffle=False)

    # Model setup
    model = TripletNet(EmbeddingNet(n_channels=6))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TripletLoss(margin=0.5)

    # Training loop
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
