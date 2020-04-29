# Project: https://github.com/adambielski/siamese-triplet
# Author: Adam Bielski https://github.com/adambielski
# License: BSD 3-Clause


import torch
import numpy as np
from matplotlib import pyplot as plt


def fit(train_loader,
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        cuda,
        log_interval,
        metrics=[],
        plot_loss_curve=True,
        plot_interval=7,
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    train_losses = []
    val_losses = []

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        train_losses.append(train_loss)

        message = f'Epoch: {epoch + 1}/{n_epochs}. Train set: Average loss: {train_loss:.4f}'
        for metric in metrics:
            message += f'\t{metric.name()}: {metric.value()}'

        # Validation stage
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        message += f'\nEpoch: {epoch + 1}/{n_epochs}. Validation set: Average loss: {val_loss:.4f}'
        for metric in metrics:
            message += f'\t{metric.name()}: {metric.value()}'

        print(message)

        if plot_loss_curve and (epoch + 1) % plot_interval == 0:
            plt.plot(train_losses, label='train')
            plt.plot(val_losses, label='val')
            plt.legend(loc='upper right')
            plt.show()

    return train_losses, val_losses


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = f'Train: [{batch_idx * len(data[0])}/{len(train_loader.dataset)}' \
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t' \
                      f'Loss: {np.mean(losses):.6f}'
            for metric in metrics:
                message += f'\t{metric.name()}: {metric.value()}'

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
