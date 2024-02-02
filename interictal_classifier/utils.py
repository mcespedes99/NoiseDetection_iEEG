"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import sklearn.metrics as skm
import os


def classication_metrics(y_true: ArrayLike, y_pred_class: ArrayLike, test=False):
    """
    Args:
      y_true: Ground truth (correct) labels.
      y_pred: Array of probability estimates provided by the model.

    Missing: AUPRC https://towardsdatascience.com/imbalanced-data-stop-using-roc-auc-and-use-auprc-instead-46af4910a494
    """
    # Compute accuracy
    acc = skm.accuracy_score(y_true, y_pred_class)
    balanced_acc = skm.balanced_accuracy_score(y_true, y_pred_class)
    # Compute precision
    # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    micro_prec = skm.precision_score(
        y_true, y_pred_class, average="micro", zero_division=0.0
    )
    macro_prec = skm.precision_score(
        y_true, y_pred_class, average="macro", zero_division=0.0
    )
    weighted_prec = skm.precision_score(
        y_true, y_pred_class, average="weighted", zero_division=0.0
    )
    # Recall
    micro_recall = skm.recall_score(y_true, y_pred_class, average="micro")
    macro_recall = skm.recall_score(y_true, y_pred_class, average="macro")
    weighted_recall = skm.recall_score(y_true, y_pred_class, average="weighted")
    # F1
    micro_f1 = skm.f1_score(y_true, y_pred_class, average="micro")
    macro_f1 = skm.f1_score(y_true, y_pred_class, average="macro")
    weighted_f1 = skm.f1_score(y_true, y_pred_class, average="weighted")
    # Assign to list
    metrics_out = [
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

    if test:
        metrics_out.append(skm.classification_report(y_true, y_pred_class))
    return metrics_out


def print_key_metrics(
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
    class_report=None,
):
    print(
        "\n-------------------- Key Metrics --------------------", end="\n", flush=True
    )
    print("\nAccuracy: {:.2f}".format(acc), end="\n", flush=True)
    print("Balanced Accuracy: {:.2f}\n".format(balanced_acc), end="\n", flush=True)

    print("Micro Precision: {:.2f}".format(micro_prec), end="\n", flush=True)
    print("Micro Recall: {:.2f}".format(micro_recall), end="\n", flush=True)
    print("Micro F1-score: {:.2f}\n".format(micro_f1), end="\n", flush=True)

    print("Macro Precision: {:.2f}".format(macro_prec), end="\n", flush=True)
    print("Macro Recall: {:.2f}".format(macro_recall), end="\n", flush=True)
    print("Macro F1-score: {:.2f}\n".format(macro_f1), end="\n", flush=True)

    print("Weighted Precision: {:.2f}".format(weighted_prec), end="\n", flush=True)
    print("Weighted Recall: {:.2f}".format(weighted_recall), end="\n", flush=True)
    print("Weighted F1-score: {:.2f}".format(weighted_f1), end="\n", flush=True)

    if class_report:
        print(
            "\n--------------- Classification Report ---------------\n",
            end="\n",
            flush=True,
        )
        print(class_report, end="\n", flush=True)

def classication_metrics_binary(y_true: ArrayLike, y_pred_class: ArrayLike):
    """
    Args:
      y_true: Ground truth (correct) labels.
      y_pred: Array of probability estimates provided by the model.

    Missing: AUPRC https://towardsdatascience.com/imbalanced-data-stop-using-roc-auc-and-use-auprc-instead-46af4910a494
    """
    # Compute accuracy
    acc = skm.accuracy_score(y_true, y_pred_class)
    balanced_acc = skm.balanced_accuracy_score(y_true, y_pred_class)
    # Compute precision
    # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    prec = skm.precision_score(y_true, y_pred_class, average="binary", zero_division=0.0)
    # Recall
    recall = skm.recall_score(y_true, y_pred_class, average="binary", zero_division=0.0)
    # F1
    f1 = skm.f1_score(y_true, y_pred_class, average="binary", zero_division=0.0)
    # Assign to list
    metrics_out = [
        acc,
        balanced_acc,
        prec,
        recall,
        f1
    ]
    return metrics_out


def print_key_metrics_binary(
    acc,
    balanced_acc,
    prec,
    recall,
    f1
):
    print(
        "\n-------------------- Key Metrics --------------------", end="\n", flush=True
    )
    print("\nAccuracy: {:.2f}".format(acc), end="\n", flush=True)
    print("Balanced Accuracy: {:.2f}\n".format(balanced_acc), end="\n", flush=True)

    print("Precision: {:.2f}".format(prec), end="\n", flush=True)
    print("Recall: {:.2f}".format(recall), end="\n", flush=True)
    print("F1-score: {:.2f}\n".format(f1), end="\n", flush=True)


# Custom transform
class RandomMask(object):
    """Masks randomly the spectrogram in a sample.
    3 masks in freq and 2 in time.
    https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-022-01942-2

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(
        self,
        p: float,
        f_max_size: int = 10,
        t_max_size: int = 10,
        n_masks_f: int = 3,
        n_masks_t: int = 3,
    ):
        assert isinstance(p, (float, int)) & (p <= 1) & (p >= 0)
        self.p = p
        self.f_max = f_max_size
        self.t_max = t_max_size
        self.n_masks_f = n_masks_f
        self.n_masks_t = n_masks_t

    def __call__(self, image):
        # print(image.shape)
        h, w = image.shape[-2:]

        image_out = np.copy(image)
        # Execute given a certain probability
        if np.random.random() < self.p:
            # Masks in freq
            for i in range(self.n_masks_f):
                # Mask goes from f_0 to f_0+f
                # print(self.f_max)
                if self.f_max > 1:
                    f = np.random.randint(low=1, high=self.f_max, size=1)[0]
                else:
                    f = 1
                f_0 = np.random.randint(low=0.0, high=h - f, size=1)[0]
                # Update image
                image_out[..., f_0 : f_0 + f, :] = 0.0
            # Masks in time
            for j in range(self.n_masks_t):
                # Mask goes from t_0 to t_0+t
                t = np.random.randint(low=1, high=self.t_max, size=1)[0]
                t_0 = np.random.randint(low=0.0, high=w - t, size=1)[0]
                # print(t_0,t)
                # Update image
                image_out[..., :, t_0 : t_0 + t] = 0.0

        return torch.from_numpy(image_out).type(torch.float)

def uncompress_zip(zip_files, out_dir):
    import zipfile
    import re
    import pandas as pd
    # Uncompress them based on the name of inst
    inst_to_zipID = dict()
    for zip_id, zip_file in enumerate(zip_files):
        # Find the institution
        with zipfile.ZipFile(zip_file, mode="r") as f:
            # Get all files
            files_zip = f.namelist()
            # Find segments.csv
            reg = re.compile("segments_new.csv")
            seg_path = list(filter(reg.search, files_zip))[0]
            # Get df
            try:
                with f.open(seg_path) as myfile:
                    df_seg = pd.read_csv(myfile, sep=",", index_col="index")
            except:
                with f.open(seg_path) as myfile:
                    df_seg = pd.read_csv(myfile, sep=",", index_col="Unnamed: 0")
            inst = df_seg["institution"].iloc[0]
            # Create output folder
            inst_dir = os.path.join(out_dir, inst)
            os.makedirs(inst_dir, exist_ok=True)
            # Uncompress
            f.extractall(inst_dir)

        


def get_tmpdir():
    if os.environ.get('SLURM_TMPDIR') != None:
        return os.environ.get('SLURM_TMPDIR')
    elif os.path.exists('/tmp'):
        return '/tmp'
    else:
        return None
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Recovered from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="",
        trace_func=print,
        restore_best_model: bool = False,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.restore_best_model = restore_best_model

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_loss, model, os.path.join(self.path, "checkpoint.pt")
            )
            # If not restore_best_model, both the checkpoint (latest model) and the best model will be saved
            if not self.restore_best_model:
                self.save_checkpoint(
                    val_loss, model, os.path.join(self.path, "best_model.pt")
                )
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
            elif not self.restore_best_model:
                self.save_checkpoint(
                    val_loss, model, os.path.join(self.path, "checkpoint.pt")
                )
        else:
            self.best_score = score
            self.save_checkpoint(
                val_loss, model, os.path.join(self.path, "checkpoint.pt")
            )
            self.counter = 0
            # If not restore_best_model, both the checkpoint (latest model) and the best model will be saved
            if not self.restore_best_model:
                self.save_checkpoint(
                    val_loss, model, os.path.join(self.path, "best_model.pt")
                )

    def save_checkpoint(self, val_loss, model, model_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss changed from ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), os.path.join(self.path, model_name))
        self.val_loss_min = val_loss


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
