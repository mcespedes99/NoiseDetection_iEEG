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

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
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
  print('train starts', end="\n", flush=True)
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0
  train_prec, train_recall, train_auc, train_f1 = 0, 0, 0, 0

  # stdout_fileno.write('dataloader \n')
  print(dataloader, end="\n", flush=True)
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
    acc, prec, recall, f1, auc = utils.classication_metrics(y, y_pred_prob)
    # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += acc
    train_prec += prec
    train_recall += recall
    train_f1 += f1
    train_auc += auc

    if batch%100 == 0:
      # stdout_fileno.write(f'Accuracy training in batch {batch}: {(y_pred_class == y).sum().item()/len(y_pred)} \n')
      print(
      f"Batch: {batch} | "
      f"acc: {acc:.4f} | "
      f"f1: {f1:.4f} | "
      f"auc: {auc:.4f} | "
    , end="\n", flush=True)
  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  train_prec = train_prec / len(dataloader)
  train_recall = train_recall / len(dataloader)
  train_f1 = train_f1 / len(dataloader)
  train_auc = train_auc / len(dataloader)
  return train_loss, (train_acc, train_prec, train_recall, train_f1, train_auc)

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
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

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0
  test_prec, test_recall, test_auc = 0, 0, 0

  # Initiate y_total and y_pred_total
  y_total = np.array([])
  y_pred_total = np.array([]).reshape(-1,4)

  # Turn on inference context manager
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
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
      y_total = np.concatenate([y_total, y])
      y_pred_total = np.concatenate([y_pred_total, y_pred_prob])

      # Calculate and accumulate accuracy
      # Not doing it per batch as each batch might not have all the data
      # y_pred_prob = torch.softmax(test_pred_logits, dim=1)
      # acc, prec, recall, f1, auc = utils.classication_metrics(y, y_pred_prob)
      # test_acc += acc
      # test_prec += prec
      # test_recall += recall
      # test_f1 += f1
      # test_auc += auc
      # test_pred_labels = test_pred_logits.argmax(dim=1)
      # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc, test_prec, test_recall, test_f1, test_auc = utils.classication_metrics(y_total, y_pred_total)
  # test_acc = test_acc / len(dataloader)
  # test_prec = test_prec / len(dataloader)
  # test_recall = test_recall / len(dataloader)
  # test_f1 = test_f1 / len(dataloader)
  # test_auc = test_auc / len(dataloader)
  return test_loss, (test_acc, test_prec, test_recall, test_f1, test_auc)

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          dir_save: str) -> Dict[str, List]:
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
  results = {"train_loss": [],
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
              "test_auc": []
  }
  
  # Make sure model on target device
  model.to(device)

  # initialize the early_stopping object
  early_stopping = utils.EarlyStopping(patience=4, verbose=True, path=os.path.join(dir_save,'checkpoint.pt'))

  # Loop through training and testing steps for a number of epochs
  # stdout_fileno.write('Epochs \n')
  print('epochs', end="\n", flush=True)
  for epoch in tqdm(range(epochs)): #tqdm(range(epochs))
    # stdout_fileno.write(f'Epoch {epoch} \n')
    # print(f'Epoch {epoch}', end="\n", flush=True)
    train_loss, train_metrics = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
    train_acc, train_prec, train_recall, train_f1, train_auc = train_metrics
    test_loss, test_metrics = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
    test_acc, test_prec, test_recall, test_f1, test_auc = test_metrics
    # Checkpoint: earlystop
    early_stopping(test_loss, model)
        
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # stdout_fileno.write out what's happening
    print(
      f"Epoch: {epoch+1} | "
      f"train_loss: {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
      f"train_f1: {train_f1:.4f} | "
      f"train_auc: {train_auc:.4f} | "
      f"test_loss: {test_loss:.4f} | "
      f"test_acc: {test_acc:.4f} \n"
      f"test_f1: {test_f1:.4f} | "
      f"test_auc: {test_auc:.4f} | "
    , end="\n", flush=True)

    # Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["train_prec"].append(train_prec)
    results["train_recall"].append(train_recall)
    results["train_f1"].append(train_f1)
    results["train_auc"].append(train_auc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    results["test_prec"].append(test_prec)
    results["test_recall"].append(test_recall)
    results["test_f1"].append(test_f1)
    results["test_auc"].append(test_auc)

  # Return the filled results at the end of the epochs
  return results
