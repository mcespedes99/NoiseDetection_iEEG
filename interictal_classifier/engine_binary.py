"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import sys
import utils
import os
import sklearn.metrics as skm
import numpy as np

def save_results(results: Dict, list_values: List):
    assert len(results) == len(list_values)
    for key, val in zip(results, list_values):
        results[key].append(val)
    return results

def merge_results(total_results_dict: Dict, tmp_results_dict):
    assert len(total_results_dict) == len(tmp_results_dict)
    for key_1, key_2 in zip(total_results_dict, tmp_results_dict):
        total_results_dict[key_1] = total_results_dict[key_1] + tmp_results_dict[key_2]
    return total_results_dict

def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    validation_freq: int,
    epoch: int,
    early_stopping,
    scheduler,
    n_iters,
    max_iter: int = None
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

    # Lists to save total metrics
    results_train = {
        "train_loss": [],
        # "train_acc": [],
        # "train_bal_acc": [],
        # "train_prec": [],
        # "train_recall": [],
        # "train_f1": []
    }
    results_val = {
        "va_loss": [],
        # "val_acc": [],
        # "val_bal_acc": [],
        # "val_prec": [],
        # "val_recall": [],
        "val_auprc": [],
    } 

    # Setup train loss and train accuracy values
    train_loss = 0
    # t_acc, bal_acc, prec, recall, f1
    # train_metrics = np.array([0,0,0,0,0])

    # stdout_fileno.write('dataloader \n')
    print(train_dataloader, end="\n", flush=True)
    # Counter to average when not sufficient samples to calculate auc
    acum_batches = 1
    y_total = np.array([])
    y_pred_total = np.array([])

    n_batches = 0
    # validation_freq = int(len(train_dataloader)*validation_freq)
    print(f'Using a validation frequency of {validation_freq}\n', flush=True)
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(train_dataloader):
        n_iters += 1
        n_batches += 1
        # Put model in train mode
        model.train()
        # Send data to target device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X).squeeze()
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred.type(torch.float), y.type(torch.float))
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        acum_batches += 1
        
        # Validate
        if batch % validation_freq == 0 and n_iters != 1:
            # Loss
            train_loss = train_loss/n_batches
            train_metrics = [train_loss]#+train_metrics
            # Append to global results
            results_train = save_results(results_train, train_metrics)
            # Print Loss
            print(f"\nMetrics batch {batch}, epoch {epoch}", end="\n", flush=True)
            print(f'\n Training loss: {train_loss:.3f}\n', end="\n", flush=True)
            # Reset metrics
            train_loss = 0
            # t_acc, bal_acc, prec, recall, f1
            # train_metrics = np.array([0,0,0,0,0])
            # y_total = np.array([])
            # y_pred_total = np.array([])
            acum_batches = 1
            n_batches = 0


            # Validation
            if test_dataloader is not None:
                val_loss, metric = test_step(
                    model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
                )
                # Print them
                # print(f"\nVal metrics batch {batch}, epoch {epoch}", end="\n", flush=True)
                # # stdout_fileno.write(f'Accuracy training in batch {batch}: {(y_pred_class == y).sum().item()/len(y_pred)} \n')
                # utils.print_key_metrics_binary(
                #     *val_metrics
                # )
                # Merge them
                val_metrics =  [val_loss, metric]#+val_metrics
                results_val = save_results(results_val, val_metrics)

                # Print Loss
                print(f'\n Validation loss: {val_loss:.3f}\n', end="\n", flush=True)

                # Change scheduler
                scheduler.step()
                print(f'\n Learning rate: {optimizer.param_groups[0]["lr"]}\n', end="\n", flush=True)

                # Checkpoint: earlystop
                early_stopping(metric, model)

            # Training can stop for 2 reasons: early stop or max_iter reached
            if (max_iter is not None) and (n_iters >= max_iter):
                early_stopping.early_stop = True
            if early_stopping.early_stop:
                print("Early stopping")
                break
    # print(t_metrics)
    return results_train, results_val, early_stopping.early_stop, n_iters


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
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X).squeeze()

            # 2. Calculate and accumulate loss
            if len(y)>0:
                loss = loss_fn(test_pred_logits.type(torch.float), y.type(torch.float))
                test_loss += loss.item()

            # Apply softmax
            y_pred_prob = torch.sigmoid(test_pred_logits)

            # # Save results
            y_total = np.concatenate([y_total, y.detach().numpy()])
            y_pred_total = np.concatenate(
                [y_pred_total, y_pred_prob.detach().numpy()]
            )

    # Check metrics
    # PR AUC
    prec, recall, _ = skm.precision_recall_curve(y_true=y_total, probas_pred=y_pred_total)
    auprc = skm.auc(recall, prec)
    print(f'\n Validation auprc: {auprc:.3f}\n', end="\n", flush=True)
    #ROC AUC
    auroc = skm.roc_auc_score(y_true=y_total, y_score=y_pred_total)
    # Print them
    print(f'\n Validation auprc: {auprc:.3f}\n', end="\n", flush=True)
    print(f'\n Validation auroc: {auroc:.3f}\n', end="\n", flush=True)
    # t_metrics = utils.classication_metrics_binary(y_total, y_pred_total)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    return test_loss, auprc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    dir_save: str,
    validation_freq: int,
    restore_best_model: bool = False,
    max_iter: int = None
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
    # Create empty results dictionary
    results_train = {
        "train_loss": [],
        # "train_acc": [],
        # "train_bal_acc": [],
        # "train_prec": [],
        # "train_recall": [],
        # "train_f1": []
    }
    results_val = {
        "va_loss": [],
        # "val_acc": [],
        # "val_bal_acc": [],
        # "val_prec": [],
        # "val_recall": [],
        "val_auprc": [],
    } 

    # Make sure model on target device
    model.to(device)

    # initialize the early_stopping object
    early_stopping = utils.EarlyStopping(
        patience=5,
        verbose=True,
        path=dir_save,
        restore_best_model=restore_best_model,
        objective='max' # as using f1
    )

    # Loop through training and testing steps for a number of epochs
    n_iters = 0
    # stdout_fileno.write('Epochs \n')
    print("epochs", end="\n", flush=True)
    for epoch in tqdm(range(epochs)):  # tqdm(range(epochs))
        # stdout_fileno.write(f'Epoch {epoch} \n')
        # print(f'Epoch {epoch}', end="\n", flush=True)
        results_train_tmp, results_val_tmp, early_stop, n_iters = train_step(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            validation_freq=validation_freq,
            epoch=epoch,
            early_stopping=early_stopping,
            scheduler = scheduler,
            n_iters=n_iters,
            max_iter=max_iter
        )

        # # loads the last checkpoint with the best model if restore_best_model
        # if restore_best_model:
        #     model.load_state_dict(torch.load(os.path.join(dir_save, "checkpoint.pt")))

        # stdout_fileno.write out what's happening

        # Update results dictionary
        results_train = merge_results(results_train, results_train_tmp)
        results_val = merge_results(results_val, results_val_tmp)

        if early_stop:
            print("Early stopping")
            break
        
    # Return the filled results at the end of the epochs
    return {**results_train, **results_val}, n_iters
