"""Include train/validation loop."""


import collections

import numpy as np
import torch
import tqdm


from calculate_metrics import calculate_accuracy


def fit_epoch(model, train_dataloader, criterion, optimizer, epoch, device):
    """Train model on current epoch.

    Args:
        model (): Model MUST be already on device
        train_dataloader ():
        criterion ():
        optimizer ():
        epoch (int): number of current epoch
        device (): torch.cuda.device("cuda" if torch.cuda.is_available() else "cpu")

    Returns:
        train_loss (float): mean train loss on current epoch,
        y_preds: (torch.tensor),
        y_true:
    """
    pbar = tqdm.tqdm(
        enumerate(train_dataloader), total=len(train_dataloader), leave=False
    )
    pbar.set_description(f"Epoch {epoch}")

    model.train()

    running_loss = 0.0
    processed_size = 0
    y_preds = []
    y_true = []

    for i, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        if len(outputs.size()) == 2:
            loss = criterion(outputs[:, 1], labels)
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().cpu().item()
        processed_size += inputs.size(0)  # !!! because batch_first=True

        y_preds.append(outputs.detach().cpu())
        y_true.append(labels.cpu())

    train_loss = running_loss / processed_size
    return train_loss, torch.cat(y_preds, dim=0), torch.cat(y_true, dim=0)


def eval_epoch(model, val_dataloader, criterion, epoch, device):
    """Validate model on current epoch.

    Args:
        model (): Model MUST be already on device
        val_dataloader ():
        criterion ():
        epoch ():
        device ():

    Returns:
        val_loss,
        y_preds,
        y_true
    """
    pbar = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False)
    pbar.set_description(f"Epoch {epoch}")

    model.eval()

    running_loss = 0.0
    processed_size = 0
    y_preds = []
    y_true = []

    for i, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            if len(outputs.size()) == 2:
                loss = criterion(outputs[:, 1], labels)
            else:
                loss = criterion(outputs, labels)

        running_loss += loss.cpu().item()
        processed_size += inputs.size(0)  # because batch_first=True

        y_preds.append(outputs.cpu())
        y_true.append(labels.cpu())

    val_loss = running_loss / processed_size
    return val_loss, torch.cat(y_preds, dim=0), torch.cat(y_true, dim=0)


def print_train_val_results(
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics_history,
    val_metrics_history,
):
    """Print loss and metrics results after each epoch during train/validation
    model.

    Args:
        epoch (int): number of current epoch
        train_loss (float):
        val_loss (float):
        train_metrics_history ():
        val_metrics_history ():

    Returns:
        None
    """
    title = "   EPOCHS    |         TRAIN         |      VALIDATION       "
    if epoch == 1:
        print(title)
        print("-" * len(title))

    print(
        " Epoch: {0:4} | Loss {1: 16} | Loss {2:16}".format(epoch, train_loss, val_loss)
    )
    for key in train_metrics_history:
        print(
            "             | {0:16} {1:4} | {2:16} {3:4}".format(
                key, train_metrics_history[key][-1], key, val_metrics_history[key][-1]
            )
        )
    print("-" * len(title))


def train_val_loop(
    model,
    opt,
    loss_func,
    train_dataloader,
    val_dataloader,
    device,
    metrics={"accuracy": calculate_accuracy},
    logits=True,
    lr_scheduler=None,
    max_epochs=100,
    patience=20,
):
    """Train, validation loop with early-stopping methodology. Saves the best
    model on validation loss result.

    Args:
        model (): Model MUST be already on device
        opt (): Optimizer to optimize loss_func
        loss_func ():
        train_dataloader ():
        val_dataloader ():
        device (): torch.cuda.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics (dict["name_of_metric": function_to_calculate_metric]): Dict with ONE or MORE key: value, where
        key - name of metric, value - function to calculate this metric
        logits (bool): Depends on model architecture. True, if model return logits or False, if model return probability
        lr_scheduler (): Learning rate scheduler
        max_epochs (int): Number of epochs to train, validate model
        patience (int): Number of epochs to stop train model, when validation loss is increasing - early-stopping
                        methodology

    Returns:
        train_loss_history,
        val_loss_history,
        train_metrics_history,
        val_metrics_history,
        model
    """
    min_loss = np.inf
    cur_patience = 0

    train_loss_history = []
    val_loss_history = []
    train_metrics_history = collections.defaultdict(list)
    val_metrics_history = collections.defaultdict(list)

    for epoch in range(1, max_epochs + 1):

        train_loss, y_preds, y_true = fit_epoch(
            model, train_dataloader, loss_func, opt, epoch, device
        )
        train_loss_history.append(train_loss)
        for metric, calculate_metric in metrics.items():
            metric_ = calculate_metric(y_preds, y_true, logits=logits)
            train_metrics_history[metric].append(metric_)

        val_loss, y_preds, y_true = eval_epoch(
            model, val_dataloader, loss_func, epoch, device
        )
        val_loss_history.append(val_loss)
        for metric, calculate_metric in metrics.items():
            metric_ = calculate_metric(y_preds, y_true, logits=logits)
            val_metrics_history[metric].append(metric_)

        print_train_val_results(
            epoch, train_loss, val_loss, train_metrics_history, val_metrics_history
        )

        if val_loss < min_loss:
            min_loss = val_loss
            best_model = model.state_dict()
            print("Save best model(Epoch: {})".format(epoch))
        else:
            cur_patience += 1
            if cur_patience == patience:
                # cur_patience = 0
                break

        if lr_scheduler is not None:
            lr_scheduler.step()

    model.load_state_dict(best_model)

    return (
        train_loss_history,
        val_loss_history,
        train_metrics_history,
        val_metrics_history,
        model,
    )


if __name__ == "__main__":
    pass
