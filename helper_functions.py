import torch
from torch import nn
from timeit import default_timer as d_timer

"""
Script that contains several useful helper functions
that were originally implemented in the base Digit Recognizer model

"""

def train_time(start: float, end: float):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time: {total_time:.3f} seconds")
    return total_time

def accuracy_fn(actual_labels, pred_labels):
    """Calculates accuracy between actual labels and prediction labels.

    Args:
        actual_labels (torch.Tensor): Actual labels from the dataset, in other words the "correct answer".
        pred_labels (torch.Tensor): Predictions to be compared to actual labels.

    Returns:
        [torch.float]: Accuracy value between actual_labels and pred_labels
    """
    correct = torch.eq(actual_labels, pred_labels).sum().item()
    return (correct / len(pred_labels)) * 100

def train_batches(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
  train_loss, train_acc = 0,0
  model.to(device)
  for batch, (X,y) in enumerate(data_loader):
    X,y = X.to(device), y.to(device)
    pred_labels = model(X)

    loss = loss_fn(pred_labels,y)
    acc = accuracy_fn(y, pred_labels.argmax(dim=1)) #converting logit to pred_probs when calc acc
    #accumulate the loss and acc for each batch
    train_loss += loss
    train_acc += acc

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  #loss and accuracy per epoch
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_batches(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
  test_loss, test_acc = 0,0
  model.to(device)
  model.eval() #put model on eval mode since its on train mode by default
  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device), y.to(device)

      test_pred = model(X)

      test_loss += loss_fn(test_pred,y)
      test_acc += accuracy_fn(actual_labels=y,
                              pred_labels=test_pred.argmax(dim=1)) #logits to pred probs
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def make_preds(model: torch.nn.Module,data:list,device: torch.device):
  pred_probs = []
  model.eval()# turn on eval mode
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample,dim=0).to(device) #adding an extra dimension to sample data

      #forward pass
      pred_raw = model(sample) # raw logits

      pred_prob = torch.softmax(pred_raw.squeeze(),dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

      pred_probs.append(pred_prob.cpu()) # get pred_prob of gpu if it is on gpu
  return  torch.stack(pred_probs)