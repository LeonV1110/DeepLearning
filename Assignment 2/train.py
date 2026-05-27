import torch


def _unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
    else:
        x, y, _ = batch

    return x, y


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    use_amp = str(device).startswith("cuda")

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(x)
                loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
        else:
            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
