from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np

from train import train_one_epoch
from evaluate import evaluate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_grid_search(model_class, param_grid, train_loader, num_classes, epochs=15, batch_size=64, k_folds=4,):

    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))

    best_acc = 0.0
    best_params = None

    dataset = train_loader.dataset

    # Extract labels once
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    skf = StratifiedKFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=42,
    )

    for values in combinations:

        params = dict(zip(keys, values))

        print("\n====================================")
        print("Testing parameters:")
        print(params)
        print("====================================")

        fold_val_accs = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(dataset)), labels)
        ):

            fold_train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=batch_size,
                shuffle=True,
            )

            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=batch_size,
                shuffle=False,
            )

            # Create model dynamically
            model_params = {
                k: v for k, v in params.items()
                if k != "learning_rate"
            }

            model = model_class(
                num_classes=num_classes,
                **model_params
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
                    f"Fold {fold+1} | "
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Val Acc: {val_acc:.2f}%"
                )

            fold_val_accs.append(val_acc)

        mean_val_acc = float(np.mean(fold_val_accs))

        print(f"\nMean CV Accuracy: {mean_val_acc:.2f}%")

        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            best_params = params

    print("\n====================================")
    print("BEST PARAMETERS")
    print("====================================")

    print(best_params)
    print(f"Best Accuracy: {best_acc:.2f}%")

    return best_params, best_acc