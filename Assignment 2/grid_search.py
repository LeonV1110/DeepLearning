from itertools import product
import torch
import torch.nn as nn

from tcn_model import MEGTCN
from train import train_one_epoch
from evaluate import evaluate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

param_grid = {
    "learning_rate": [1e-3, 5e-4],
    "kernel_size": [3, 7, 15],
    "dropout": [0.2, 0.5],
    "hidden_channels": [32, 64],
}

def run_grid_search(train_loader, test_loader, num_classes, epochs=15):

    keys = param_grid.keys()

    combinations = list(product(*param_grid.values()))

    best_acc = 0.0
    best_params = None

    for values in combinations:

        params = dict(zip(keys, values))

        print("\n====================================")
        print("Testing parameters:")
        print(params)
        print("====================================")

        model = MEGTCN(
            num_classes=num_classes,
            hidden_channels=params["hidden_channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
        )

        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
            )

            val_loss, val_acc = evaluate(
                model,
                test_loader,
                criterion,
                DEVICE,
            )

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}%"
            )

        if val_acc > best_acc:

            best_acc = val_acc
            best_params = params

    print("\n====================================")
    print("BEST PARAMETERS")
    print("====================================")

    print(best_params)

    print(f"Best Accuracy: {best_acc:.2f}%")