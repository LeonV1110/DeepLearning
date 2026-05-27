from itertools import product
import torch
import torch.nn as nn

from tcn_model import MEGTCN
from train import train_one_epoch
from evaluate import evaluate
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

param_grid = {
    "learning_rate": [1e-3, 5e-4],
    "kernel_size": [3, 7, 15],
    "dropout": [0.2, 0.5],
    "hidden_channels": [32, 64],
}


def run_grid_search(train_loader, num_classes, epochs=15, batch_size=64, k_folds=4):

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

        # K-fold cross-validation with stratification
        dataset = train_loader.dataset

        # Get labels from dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_val_accs = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(dataset)), labels)
        ):

            fold_train_loader = DataLoader(
                Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
            )

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

            print(f"Starting fold {fold+1}/{k_folds}")

            for epoch in range(epochs):

                train_loss, train_acc = train_one_epoch(
                    model,
                    fold_train_loader,
                    criterion,
                    optimizer,
                    DEVICE,
                )

                val_loss, val_acc = evaluate(
                    model,
                    val_loader,
                    criterion,
                    DEVICE,
                )

                print(
                    f"Fold {fold+1} | Epoch {epoch+1}/{epochs} | "
                    f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
                )

            fold_val_accs.append(val_acc)

        # average validation acc across folds
        mean_val_acc = float(np.mean(fold_val_accs))

        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            best_params = params

    print("\n====================================")
    print("BEST PARAMETERS")
    print("====================================")

    print(best_params)

    print(f"Best Accuracy: {best_acc:.2f}%")
