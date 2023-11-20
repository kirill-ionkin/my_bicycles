def fit_epoch(model, train_dataloader, criterion, optimizer, epoch, device):
    """"""
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
    """"""
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
    """"""
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

        if val_loss < min_loss:
            min_loss = val_loss
            best_model = model.state_dict()
            print(f"Save best model(Epoch: {epoch})")
        else:
            cur_patience += 1
            if cur_patience == patience:
                cur_patience = 0
                break

        if lr_scheduler is not None:
            lr_scheduler.step()

        print(
            f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
        )
        print(
            f'---------  Training accuracy: {train_metrics_history["accuracy"][-1]}, Validation accuracy: {val_metrics_history["accuracy"][-1]}'
        )

    model.load_state_dict(best_model)

    return (
        train_loss_history,
        val_loss_history,
        train_metrics_history,
        val_metrics_history,
        model,
    )


if __name__ != "__main__":
    import collections

    import numpy as np

    import torch

    import tqdm

    from calculate_metrics import calculate_accuracy
