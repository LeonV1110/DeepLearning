from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np

from train import train_one_epoch
from evaluate import evaluate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_grid_search(model_class, param_grid, train_loader, num_classes, epochs=15, k_folds=4,):

    keys = param_grid.keys()
    combinations = list(product(*param_grid.values()))

    top_models = []

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
                batch_size=params["batch_size"],
                shuffle=True,
            )

            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=params["batch_size"],
                shuffle=False,
            )

            # Create model dynamically
            model_params = {
                k: v for k, v in params.items()
                if k not in ["learning_rate", "batch_size"]
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

        top_models.append(
        {
            "params": params.copy(),
            "accuracy": mean_val_acc,
        }
    )

    top_models = sorted(
        top_models,
        key=lambda x: x["accuracy"],
        reverse=True,
    )

    top_models = top_models[:5]

    print("\nTOP 5 CONFIGURATIONS")

    for rank, result in enumerate(top_models, start=1):
        print(
            f"{rank}. "
            f"Accuracy={result['accuracy']:.2f}% "
            f"Params={result['params']}"
        )

    return top_models

def run_person_grid_search(model_class, param_grid, person_splits, num_classes, epochs=15):
    """
    Grid search using person-level cross-validation.
    Each fold = leave-one-person-out.
    """

    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    top_models = []

    print(f"Total configs: {len(combinations)}")

    for values in combinations:

        params = dict(zip(keys, values))

        print("\n====================================")
        print("Testing parameters:")
        print(params)
        print("====================================")

        fold_accuracies = []

        #PERSON-LEVEL CROSS VALIDATION
        for fold, (test_name, test_subset, test_person_id) in enumerate(person_splits):

            # train = all other persons
            train_subsets = [subset for split_name, subset, _ in person_splits if split_name != test_name]

            train_dataset = ConcatDataset(train_subsets)

            train_loader = DataLoader(
                train_dataset,
                batch_size=params["batch_size"],
                shuffle=True,
                pin_memory=(DEVICE == "cuda"),
            )

            val_loader = DataLoader(
                test_subset,
                batch_size=params["batch_size"],
                shuffle=False,
                pin_memory=(DEVICE == "cuda"),
            )

            # Create model dynamically
            model_params = {
                k: v for k, v in params.items()
                if k not in ["learning_rate", "batch_size"]
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

            print(f"\nFold {fold+1}/{len(person_splits)} | Test person: {test_person_id}")

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
                    val_loader,
                    criterion,
                    DEVICE,
                )

                print(
                    f"Fold {fold+1} | Epoch {epoch+1}/{epochs} | "
                    f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
                )

            fold_accuracies.append(val_acc)

        mean_acc = float(np.mean(fold_accuracies))

        print(f"\nMean person CV Accuracy: {mean_acc:.2f}%")

        top_models.append(
        {
            "params": params.copy(),
            "accuracy": mean_acc,
        }
    )

    top_models = sorted(
        top_models,
        key=lambda x: x["accuracy"],
        reverse=True,
    )

    top_models = top_models[:5]

    print("\nTOP 5 CONFIGURATIONS")

    for rank, result in enumerate(top_models, start=1):
        print(
            f"{rank}. "
            f"Accuracy={result['accuracy']:.2f}% "
            f"Params={result['params']}"
        )

    return top_models