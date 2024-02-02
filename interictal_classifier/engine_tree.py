"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import sys
import utils
import os
from xgboost import XGBClassifier
import numpy as np
import sklearn.metrics as skm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dir_save: str,
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

    # Setup train loss and train accuracy values
    train_loss = 0
    t_acc, t_bal_acc = 0, 0
    t_micro_prec, t_macro_prec, t_weighted_prec = 0, 0, 0
    t_micro_recall, t_macro_recall, t_weighted_recall = 0, 0, 0
    t_micro_f1, t_macro_f1, t_weighted_f1 = 0, 0, 0

    # stdout_fileno.write('dataloader \n')
    y_total = np.array([])
    acum_batches = 1
    print(dataloader, end="\n", flush=True)
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        y_total = np.concatenate((y_total, y.detach().numpy()))
        if len(np.unique(y_total)) == 4:
            pretrained_path = None
            if os.path.exists(os.path.join(dir_save, "model.model")):
                pretrained_path = os.path.join(dir_save, "model.model")
            # Fit
            model.fit(X.detach().numpy(), y.detach().numpy(), xgb_model=pretrained_path)
            # Save model
            model.save_model(os.path.join(dir_save, "model.model"))
            # Predict
            y_pred = model.predict(X.detach().numpy())

            # Check metrics
            metrics_train = utils.classication_metrics(y.detach().numpy(), y_pred)
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
            acum_batches = 1
        else:
            acum_batches += 1
        if batch % 20 == 0:
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

        # if batch == 100:
        #   break
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    t_metrics = [
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
    ]
    unused_batches = 0
    if acum_batches > 1:
        unused_batches = acum_batches
    for t_metric in t_metrics:
        t_metric = t_metric / (len(dataloader) - unused_batches)
    return train_loss, t_metrics


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
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

    # Initiate values
    y_total, y_pred_total = np.array([]), np.array([])

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            y_pred = model.predict(X.detach().numpy())

            # Save results
            y_total = np.concatenate((y_total, y.detach().numpy()))
            y_pred_total = np.concatenate((y_pred_total, y_pred), axis=0)

    # Check metrics
    t_metrics = utils.classication_metrics(y_total, y_pred_total, test=True)
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    return test_loss, t_metrics


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    dir_save: str,
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
    # print('Before loading data\n', end="\n", flush=True)
    # X, y = next(iter(train_dataloader))
    # print('Size train set',X.shape, y.shape, end="\n", flush=True)
    # X_train = X.detach().numpy()
    # y_train = y.detach().numpy()
    # X, y = next(iter(test_dataloader))
    # print('Size val set',X.shape, y.shape, end="\n", flush=True)
    # X_val = X.detach().numpy()
    # y_val = y.detach().numpy()

    # model = XGBClassifier(
    #     max_depth=5,  # Depth of each tree
    #     learning_rate=0.1,            # How much to shrink error in each subsequent training. Trade-off with no. estimators.
    #     n_estimators=2000,  # How many trees to use, the more the better, but decrease learning rate if many used.
    #     verbosity=1,  # If to show more errors or not.
    #     objective="multi:softmax",  # Type of target variable.
    #     booster="gbtree",  # What to boost. Trees in this case.
    #     n_jobs=32,  # Parallel jobs to run. Set your processor number.
    #     random_state=20201107,  # Seed
    #     eval_metric='f1'
    # )

    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=10, verbose=True)
    
    # # Predict 
    # y_pred = model.predict(X_val)
    # t_metrics = utils.classication_metrics(y_val, y_pred, test=True)
    # print('\nTest Metrics:\n')
    # utils.print_key_metrics(*t_metrics)
    # # initialize the early_stopping object
    # early_stopping = utils.EarlyStopping(
    #     patience=4, verbose=True, path=os.path.join(dir_save, "checkpoint.pt")
    # )

    # Loop through training and testing steps for a number of epochs
    # stdout_fileno.write('Epochs \n')
    print("epochs", end="\n", flush=True)
    for epoch in tqdm(range(epochs)):  # tqdm(range(epochs))
        # stdout_fileno.write(f'Epoch {epoch} \n')
        # print(f'Epoch {epoch}', end="\n", flush=True)
        train_loss, train_metrics = train_step(
            model=model, dataloader=train_dataloader, dir_save=dir_save
        )
        test_loss, test_metrics = test_step(model=model, dataloader=test_dataloader)
        # Checkpoint: earlystop
        # early_stopping(test_loss, model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        # load the last checkpoint with the best model
        # model.load_state_dict(torch.load(os.path.join(dir_save, "checkpoint.pt")))

        # Print what's happening
        print("\nTrain Metrics:")
        utils.print_key_metrics(*train_metrics)
        print("\nTest Metrics:")
        utils.print_key_metrics(*test_metrics)

        # Update results dictionary
        # results["train_loss"].append(train_loss)
        # results["train_acc"].append(train_acc)
        # results["train_prec"].append(train_prec)
        # results["train_recall"].append(train_recall)
        # results["train_f1"].append(train_f1)
        # results["train_auc"].append(train_auc)
        # results["test_loss"].append(test_loss)
        # results["test_acc"].append(test_acc)
        # results["test_prec"].append(test_prec)
        # results["test_recall"].append(test_recall)
        # results["test_f1"].append(test_f1)
        # results["test_auc"].append(test_auc)

    # Return the filled results at the end of the epochs
    return results
