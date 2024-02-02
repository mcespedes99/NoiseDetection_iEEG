"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import sys
import utils
import os
import numpy as np


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_classes: int
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # stdout_fileno = sys.stdout
    # stdout_fileno.write('train starts \n')
    print("train starts", end="\n", flush=True)
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0
    t_acc, t_bal_acc = 0, 0
    t_micro_prec, t_macro_prec, t_weighted_prec = 0, 0, 0
    t_micro_recall, t_macro_recall, t_weighted_recall = 0, 0, 0
    t_micro_f1, t_macro_f1, t_weighted_f1 = 0, 0, 0

    # stdout_fileno.write('dataloader \n')
    print(dataloader, end="\n", flush=True)
    # Counter to average when not sufficient samples to calculate auc
    acum_batches = 1
    y_total = np.array([])
    y_pred_total = np.array([])

    # Use only 20% of the total training data per epoch
    percentage_use = 1
    print(f'{percentage_use*100}% of the batches used per epoch.', end="\n", flush=True)
    n_batches = int(percentage_use*len(dataloader))
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # stdout_fileno.write('Init \n')
        # print('init', end="\n", flush=True)
        # Send data to target device
        X, y = X.to(device), y.to(device)
        # stdout_fileno.write('1 \n')
        # print('1', end="\n", flush=True)
        # 1. Forward pass
        y_pred = model(X)
        # stdout_fileno.write('2 \n')
        # print('2', end="\n", flush=True)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        # stdout_fileno.write('Pred \n')
        # print('Pred', end="\n", flush=True)
        # Calculate and accumulate accuracy metric across all batches
        y_pred_prob = torch.softmax(y_pred, dim=1)
        # Add to total
        # Save results
        y_total = np.concatenate([y_total, y.detach().numpy()])
        y_pred_total = np.concatenate(
            [y_pred_total, np.argmax(y_pred_prob.detach().numpy(), axis=1)]
        )
        # Check status
        if len(np.unique(y_total)) == n_classes:
            # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            # to be able to consider accumulated batches. Assigns the same metric to all acum batches
            # Check metrics
            metrics_train = utils.classication_metrics(y_total, y_pred_total)
            (
                acc,
                balanced_acc,
                micro_prec,
                macro_prec,
                weighted_prec,
                micro_recall,
                macro_recall,
                weighted_recall,
                micro_f1,
                macro_f1,
                weighted_f1,
            ) = metrics_train
            # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            # to be able to consider accumulated batches. Assigns the same metric to all acum batches
            t_acc += acc * acum_batches
            t_bal_acc += balanced_acc * acum_batches
            t_micro_prec += micro_prec * acum_batches
            t_macro_prec += macro_prec * acum_batches
            t_weighted_prec += weighted_prec * acum_batches
            t_micro_recall += micro_recall * acum_batches
            t_macro_recall += macro_recall * acum_batches
            t_weighted_recall += weighted_recall * acum_batches
            t_micro_f1 += micro_f1 * acum_batches
            t_macro_f1 += macro_f1 * acum_batches
            t_weighted_f1 += weighted_f1 * acum_batches
            # Reset variables
            y_total = np.array([])
            y_pred_total = np.array([])
            acum_batches = 1
            if batch % 200 == 0:
                print(f"\nMetrics batch {batch}")
                # stdout_fileno.write(f'Accuracy training in batch {batch}: {(y_pred_class == y).sum().item()/len(y_pred)} \n')
                utils.print_key_metrics(
                    acc,
                    balanced_acc,
                    micro_prec,
                    macro_prec,
                    weighted_prec,
                    micro_recall,
                    macro_recall,
                    weighted_recall,
                    micro_f1,
                    macro_f1,
                    weighted_f1,
                )
                # print(X.shape)
        else:
            acum_batches += 1

        if batch == n_batches:
          break
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / n_batches#len(dataloader)
    t_metrics = [
        t_acc,
        t_bal_acc,
        t_micro_prec,
        t_macro_prec,
        t_weighted_prec,
        t_micro_recall,
        t_macro_recall,
        t_weighted_recall,
        t_micro_f1,
        t_macro_f1,
        t_weighted_f1,
    ]
    unused_batches = 0
    if acum_batches > 1:
        unused_batches = acum_batches
    for t_id,t_metric in enumerate(t_metrics):
        t_metrics[t_id] = t_metric / (n_batches - unused_batches)#(len(dataloader) - unused_batches)
    return train_loss, t_metrics


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Initiate values
    y_total, y_pred_total = np.array([]), np.array([])

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # print(X.shape)
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Apply softmax
            y_pred_prob = torch.softmax(test_pred_logits, dim=1)

            # Save results
            y_total = np.concatenate([y_total, y.detach().numpy()])
            y_pred_total = np.concatenate(
                [y_pred_total, np.argmax(y_pred_prob.detach().numpy(), axis=1)]
            )

    # Check metrics
    t_metrics = utils.classication_metrics(y_total, y_pred_total, test=True)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    return test_loss, t_metrics


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    dir_save: str,
    n_classes: int,
    restore_best_model: bool = False,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, stdout_fileno.writes and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    dir_save: Path to which save the model and results to.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
            train_acc: [...],
            test_loss: [...],
            test_acc: [...]}
    For example if training for epochs=2:
            {train_loss: [2.0616, 1.0537],
            train_acc: [0.3945, 0.3945],
            test_loss: [1.2641, 1.5706],
            test_acc: [0.3400, 0.2973]}
    """
    stdout_fileno = sys.stdout
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_prec": [],
        "train_recall": [],
        "train_f1": [],
        "train_auc": [],
        "test_loss": [],
        "test_acc": [],
        "test_prec": [],
        "test_recall": [],
        "test_f1": [],
        "test_auc": [],
    }

    # Make sure model on target device
    model.to(device)

    # initialize the early_stopping object
    early_stopping = utils.EarlyStopping(
        patience=10,
        verbose=True,
        path=dir_save,
        restore_best_model=restore_best_model,
    )

    # Loop through training and testing steps for a number of epochs
    # stdout_fileno.write('Epochs \n')
    print("epochs", end="\n", flush=True)
    for epoch in tqdm(range(epochs)):  # tqdm(range(epochs))
        # stdout_fileno.write(f'Epoch {epoch} \n')
        # print(f'Epoch {epoch}', end="\n", flush=True)
        train_loss, train_metrics = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            n_classes=n_classes
        )
        # train_acc, train_prec, train_recall, train_f1, train_auc = train_metrics
        test_loss, test_metrics = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )
        # test_acc, test_prec, test_recall, test_f1, test_auc = test_metrics
        # Print what's happening
        print(f"\nEpoch {epoch}", end="\n", flush=True)
        print("\nTrain Metrics:", end="\n", flush=True)
        utils.print_key_metrics(*train_metrics)
        print("\nTest Metrics:")
        utils.print_key_metrics(*test_metrics)
        # Checkpoint: earlystop
        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # loads the last checkpoint with the best model if restore_best_model
        if restore_best_model:
            model.load_state_dict(torch.load(os.path.join(dir_save, "checkpoint.pt")))

        # stdout_fileno.write out what's happening

        # Update results dictionary
        results["train_loss"].append(train_loss)
        # results["train_acc"].append(train_acc)
        # results["train_prec"].append(train_prec)
        # results["train_recall"].append(train_recall)
        # results["train_f1"].append(train_f1)
        # results["train_auc"].append(train_auc)
        results["test_loss"].append(test_loss)
        # results["test_acc"].append(test_acc)
        # results["test_prec"].append(test_prec)
        # results["test_recall"].append(test_recall)
        # results["test_f1"].append(test_f1)
        # results["test_auc"].append(test_auc)
        print(
            f'\n Train loss history: {results["train_loss"]} \n', end="\n", flush=True
        )
        print(f'\n Test loss history: {results["test_loss"]} \n', end="\n", flush=True)
    # Return the filled results at the end of the epochs
    return results
