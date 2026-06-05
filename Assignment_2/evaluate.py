import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from dataset import SensorFilteredDataset

from train import train_one_epoch


def _unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
    else:
        x, y, _ = batch

    return x, y


def evaluate(model, loader, criterion, device):

    model.eval()
    use_amp = str(device).startswith("cuda")

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:

            x, y = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(x)
                    loss = criterion(outputs, y)
            else:
                outputs = model(x)
                loss = criterion(outputs, y)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def evaluate_top_models_cv(
    model_class,
    top_models,
    train_loader,
    num_classes,
    device,
    epochs=20,
    n_runs=5,
    k_folds=4,
):
    """
    Re-evaluate top models using repeated K-fold CV.
    """

    dataset = train_loader.dataset

    labels = np.array([
        dataset[i][1]
        for i in range(len(dataset))
    ])

    results = []

    for rank, model_info in enumerate(top_models, start=1):

        params = model_info["params"]

        print("\n" + "=" * 60)
        print(f"MODEL {rank}")
        print(params)
        print("=" * 60)

        run_accuracies = []

        for run in range(n_runs):

            seed = 2345 + run

            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            skf = StratifiedKFold(
                n_splits=k_folds,
                shuffle=True,
                random_state=seed,
            )

            fold_accs = []

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(np.zeros(len(dataset)), labels)
            ):

                train_fold_loader = DataLoader(
                    Subset(dataset, train_idx),
                    batch_size=params["batch_size"],
                    shuffle=True,
                )

                val_loader = DataLoader(
                    Subset(dataset, val_idx),
                    batch_size=params["batch_size"],
                    shuffle=False,
                )

                model_params = {
                    k: v
                    for k, v in params.items()
                    if k not in ["learning_rate", "batch_size"]
                }

                model = model_class(
                    num_classes=num_classes,
                    **model_params
                ).to(device)

                criterion = nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=params["learning_rate"],
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,
                )

                for epoch in range(epochs):

                    train_one_epoch(
                        model,
                        train_fold_loader,
                        criterion,
                        optimizer,
                        device,
                    )

                    scheduler.step()

                _, val_acc = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                fold_accs.append(val_acc)

            run_mean_acc = np.mean(fold_accs)

            run_accuracies.append(run_mean_acc)

            print(
                f"Run {run+1}/{n_runs} | "
                f"CV Accuracy: {run_mean_acc:.2f}%"
            )

        mean_acc = np.mean(run_accuracies)
        std_acc = np.std(run_accuracies)

        results.append(
            {
                "params": params,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "best_acc": np.max(run_accuracies),
                "worst_acc": np.min(run_accuracies),
            }
        )

        print(
            f"\nMean={mean_acc:.2f}% "
            f"Std={std_acc:.2f}%"
        )

    results = sorted(
        results,
        key=lambda x: x["mean_acc"],
        reverse=True,
    )

    print("\n" + "=" * 60)
    print("FINAL RANKING")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. "
            f"Mean={result['mean_acc']:.2f}% "
            f"Std={result['std_acc']:.2f}% "
            f"Params={result['params']}"
        )

    return results

def evaluate_top_models_person_cv(
    model_class,
    top_models,
    person_splits,
    num_classes,
    device,
    epochs=20,
    n_runs=5,
):
    """
    Re-evaluate top hyperparameter configurations using
    repeated leave-one-person-out cross-validation.

    Returns models ranked by mean accuracy.
    """

    results = []

    for rank, model_info in enumerate(top_models, start=1):

        params = model_info["params"]

        print("\n" + "=" * 50)
        print(f"MODEL {rank}")
        print(params)
        print("=" * 50)

        run_scores = []

        for run in range(n_runs):

            seed = 2345 + run

            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            fold_scores = []

            print(f"\nRun {run+1}/{n_runs}")

            for fold, (test_name, test_subset, test_person_id) in enumerate(person_splits):

                train_subsets = [
                    subset
                    for split_name, subset, _
                    in person_splits
                    if split_name != test_name
                ]

                train_dataset = ConcatDataset(train_subsets)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=params["batch_size"],
                    shuffle=True
                )

                val_loader = DataLoader(
                    test_subset,
                    batch_size=params["batch_size"],
                    shuffle=False
                )

                model_params = {
                    k: v
                    for k, v in params.items()
                    if k not in ["learning_rate", "batch_size"]
                }

                model = model_class(
                    num_classes=num_classes,
                    **model_params
                ).to(device)

                criterion = nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=params["learning_rate"],
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,
                )

                for epoch in range(epochs):

                    train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                    )

                    scheduler.step()

                _, val_acc = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                )

                fold_scores.append(val_acc)

                print(
                    f"  Fold {fold+1}/{len(person_splits)} "
                    f"| Person {test_person_id} "
                    f"| Acc {val_acc:.2f}%"
                )

            run_mean = np.mean(fold_scores)

            run_scores.append(run_mean)

            print(
                f"Run {run+1} Mean Person-CV Accuracy: "
                f"{run_mean:.2f}%"
            )

        mean_acc = np.mean(run_scores)
        std_acc = np.std(run_scores)

        results.append({
            "params": params,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "best_acc": np.max(run_scores),
            "worst_acc": np.min(run_scores),
            "score": mean_acc - std_acc,
        })

        print(
            f"\nFINAL FOR MODEL {rank}"
            f"\nMean Accuracy: {mean_acc:.2f}%"
            f"\nStd Accuracy: {std_acc:.2f}%"
            f"\nScore (mean-std): {mean_acc - std_acc:.2f}"
        )

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True,
    )

    print("\n" + "=" * 50)
    print("FINAL RANKING")
    print("=" * 50)

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"Mean={result['mean_acc']:.2f}% "
            f"Std={result['std_acc']:.2f}% "
            f"Score={result['score']:.2f}"
        )

        print(result["params"])

    return results

def compare_models(
    model_configs,
    train_dataset,
    test_dataset,
    num_classes,
    device,
    sensor_scenarios=None,
    epochs=20,
    n_runs=5,
):

    if sensor_scenarios is None:
        sensor_scenarios = {"all": None}

    results = []

    for config in model_configs:

        name = config["name"]
        model_class = config["model_class"]
        params = config["params"]

        print("\n" + "=" * 50)
        print(f"MODEL: {name}")
        print("=" * 50)

        model_results = {}

        #SENSOR LOOP
        for scenario_name, sensor_idx in sensor_scenarios.items():

            print("\n" + "-" * 50)
            print(f"SENSOR SET: {scenario_name}")
            print("-" * 50)

            accuracies = []

            for run in range(n_runs):

                seed = 2345 + run
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # wrap datasets if needed for sensors (top 100, top 50)
                if sensor_idx is not None:
                    train_ds = SensorFilteredDataset(train_dataset, sensor_idx)
                    test_ds = SensorFilteredDataset(test_dataset, sensor_idx)
                else:
                    train_ds = train_dataset
                    test_ds = test_dataset

                train_loader = DataLoader(
                    train_ds,
                    batch_size=params["batch_size"],
                    shuffle=True,
                )

                test_loader = DataLoader(
                    test_ds,
                    batch_size=params["batch_size"],
                    shuffle=False,
                )

                model_params = {
                    k: v for k, v in params.items()
                    if k not in ["learning_rate", "batch_size"]
                }

                sensor_count = (
                    248 if sensor_idx is None
                    else len(sensor_idx)
                )

                print(
                    f"Scenario={scenario_name}, "
                    f"Sensors={sensor_count}"
                )

                model = model_class(
                    num_classes=num_classes,
                    num_sensors=sensor_count,
                    **model_params,
                ).to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=params["learning_rate"],
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,
                )

                for epoch in range(epochs):
                    train_loss, train_acc = train_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                    )

                    scheduler.step()

                test_loss, test_acc = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device,
                )

                accuracies.append(test_acc)

                print(f"Run {run+1}/{n_runs} | Acc: {test_acc:.2f}%")

            model_results[scenario_name] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "runs": accuracies,
            }

        results.append({
            "model": name,
            "results": model_results,
        })

    #PRINT SUMMARY
    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)

    for r in results:
        print(f"\nMODEL: {r['model']}")
        for scen, stats in r["results"].items():
            print(
                f"  {scen}: "
                f"{stats['mean']:.2f}% +- {stats['std']:.2f}%"
            )

    return results