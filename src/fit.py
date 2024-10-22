import torch


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Fit function to train the model and collect loss data for plotting the learning curve.
    """
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
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

        val_loss /= len(val_loader)
        return val_loss, metrics
