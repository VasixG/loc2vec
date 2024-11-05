import torch
import matplotlib.pyplot as plt


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Fit function to train the model and collect loss data for plotting the learning curve.
    """
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, n_epochs):

        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        scheduler.step()
        train_losses.append(train_loss)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        print(message)

    return train_losses, val_losses


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics=[]):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        if cuda:
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

        optimizer.zero_grad()
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)
        loss = loss_fn(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            message = f'Train: [{batch_idx * len(anchor)}/{len(train_loader.dataset)} ' \
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
            print(message)

    total_loss /= len(train_loader)
    return total_loss, metrics


def plot_channels_separately(image, title_prefix):
    """
    Plot each channel of a given image as a separate subplot.
    """
    channels = image.shape[0]
    fig, axs = plt.subplots(1, channels, figsize=(15, 5))
    fig.suptitle(title_prefix)

    for i in range(channels):
        axs[i].imshow(image[i].cpu().numpy(), cmap='gray')
        axs[i].set_title(f"Channel {i + 1}")
        axs[i].axis('off')

    plt.show()


def plot_anchor_positive_negative(anchor, positive, negative, batch_idx):
    """
    Plot anchor, closest positive, and farthest negative examples,
    displaying each channel separately.
    """
    plot_channels_separately(anchor, f"Batch {batch_idx} - Anchor Example Channels")
    plot_channels_separately(positive, f"Batch {batch_idx} - Closest Positive Channels")
    plot_channels_separately(negative, f"Batch {batch_idx} - Farthest Negative Channels")


def test_epoch(val_loader, model, loss_fn, cuda, metrics=[]):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0

        for batch_idx, (anchor, positive, negative) in enumerate(val_loader):
            if cuda:
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            anchor_output, positive_output, negative_output = model(anchor, positive, negative)
            loss = loss_fn(anchor_output, positive_output, negative_output)
            val_loss += loss.item()

        pos_distances = torch.norm(anchor_output - positive_output, dim=1)
        neg_distances = torch.norm(anchor_output - negative_output, dim=1)
        closest_pos_idx = torch.argmin(pos_distances).item()
        farthest_neg_idx = torch.argmax(neg_distances).item()

        anchor_example = anchor[closest_pos_idx]
        closest_positive_example = positive[closest_pos_idx]
        farthest_negative_example = negative[farthest_neg_idx]

        plot_anchor_positive_negative(anchor_example, closest_positive_example,
                                      farthest_negative_example, batch_idx)

        val_loss /= len(val_loader)
        return val_loss, metrics
