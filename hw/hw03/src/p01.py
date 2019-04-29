import heapq
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from typing import Optional, Tuple

TABLES_PATH = Path("tables")


def extract_dataset(path: Path, transform_fields: Optional[dict] = None, add_id_col: bool = False):
    with open(path, "r") as f_in:
        lines = f_in.read().splitlines()
    spl = [l.split(",") for l in lines]
    labels = [l[-1] for l in spl]
    vals = torch.Tensor([[float(v.strip()) for v in l[:-1]] for l in spl])
    vals, transform_fields = normalize(vals, transform_fields)

    out_path = TABLES_PATH / (path.stem + "_normalized.tex")
    export_dataset(vals, labels, out_path, add_id_col)
    return vals, labels, transform_fields


def export_dataset(x: torch.Tensor, y, out_file: Path, add_id_col: bool = False):
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w+") as f_out:
        for r in range(x.shape[0]):
            if r > 0: f_out.write("\n")
            if add_id_col:
                f_out.write("%d & " % (r+1))
            for c in range(x.shape[1]):
                if c > 0: f_out.write(" & ")
                f_out.write("%.3f" % float(x[r, c]))
            f_out.write(r" & %s \\\hline" % y[r])


def _calc_l1_loss_tensor(train_x: torch.Tensor, sample: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    r""" Calculates L1 loss on a per feature basis """
    l1 = (train_x - sample).abs()
    return l1, torch.sum(l1, dim=1)


def normalize(vals: torch.Tensor, fields: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
    r"""
    Performs min/max normalization.  Allows specified of \p mins and \p maxs so test set has same
    normalization as training set.
    """
    if fields is None: fields = dict()
    mins_key, maxs_key = "mins", "maxs"
    if mins_key not in fields:
        fields[mins_key] = torch.min(vals, dim=0)[0]
    vals -= fields[mins_key]

    if maxs_key not in fields:
        fields[maxs_key] = torch.max(vals, dim=0)[0]
    return vals / fields[maxs_key], fields


def standardize(vals: torch.Tensor, fields: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
    r"""
    Performs min/max normalization.  Allows specified of \p mins and \p maxs so test set has same
    normalization as training set.
    """
    if fields is None: fields = dict()
    mean_key, stdev_key = "mean", "stdev"
    if mean_key not in fields:
        fields[mean_key] = torch.mean(vals, dim=0)
    vals -= fields[mean_key]

    if stdev_key not in fields:
        fields[stdev_key] = torch.std(vals, dim=0)
    return vals / fields[stdev_key], fields


def predict_part_b(train_x, train_y, unlabel_x, base_file_path: str):
    r"""
    Construct the predictions for part B

    :param train_x:
    :param train_y:
    :param unlabel_x:
    :param base_file_path: "Base file path to write the results
    """
    pred = []
    for i in range(unlabel_x.shape[0]):
        l1, tot = _calc_l1_loss_tensor(train_x, unlabel_x[i])
        best = torch.argmin(tot)
        pred.append(train_y[best])

        path = Path(str(base_file_path) + "p" + str(i+1) + "_err.tex")
        path.parent.mkdir(exist_ok=True)
        with open(path, "w+") as f_out:
            for r in range(l1.shape[0]):
                if r > 0: f_out.write("\n")
                f_out.write("%d & " % (r + 1))
                for c in range(l1.shape[1]):
                    f_out.write("%.3f & " % l1[r, c])
                if best == r: f_out.write(r"\textbf{")
                f_out.write("%.3f" % tot[r])
                if best == r: f_out.write(r"}")
                f_out.write(r"\\\hline")

    with open(str(base_file_path) + "pred.tex", "w+") as f_out:
        for i, label in enumerate(pred):
            f_out.write(r"%d & %s \\\hline" % (i + 1, label))


def get_label(k: int, train_x: torch.Tensor, train_y, sample: torch.Tensor):
    r"""
    Calculates the label for \p sample and returns the label.
    :param k: k for NN
    :param train_x:
    :param train_y:
    :param sample:
    :return: Predicted label
    """
    _, tot = _calc_l1_loss_tensor(train_x, sample)
    h = []
    for i in range(train_x.shape[0]):
        l = tot[i]
        if len(h) < k or abs(h[0][0]) > l:
            if len(h) == k: heapq.heappop(h)
            heapq.heappush(h, (-l, i))

    # Get the most common label
    cntr = Counter([train_y[r] for _, r in h])
    label, cnt = Counter(cntr).most_common(1)[0]
    # Break ties to "BAD"
    if cnt + cnt == k: label = "BAD"
    return label, h


def get_best_k(num_train: int, train_x, train_y) -> int:
    r""" Calculates the best k """
    # Split the training set into validation and test
    train_sub_x, train_sub_y = train_x[:num_train], train_y[:num_train]
    valid_x, valid_y = train_x[num_train:], train_y[num_train:]

    # Export the validation set losses
    TABLES_PATH.mkdir(exist_ok=True)
    for i in range(valid_x.shape[0]):
        id_num = i + num_train
        loss, tot = _calc_l1_loss_tensor(train_x, valid_x[i])
        with open(TABLES_PATH / ("p01_c_id_%d_loss.tex" % (id_num + 1)), "w+") as f_out:
            for r in range(num_train):
                if r > 0: f_out.write("\n")
                f_out.write("%d & " % (r + 1))
                for c in range(train_x.shape[1]):
                    f_out.write("%.3f & " % (loss[r, c]))
                f_out.write(r"%.3f\\\hline" % (tot[r]))

    accuracy = []
    for k in range(1, num_train + 1):
        num_correct = 0
        labels, selected = [], []
        for i in range(0, valid_x.shape[0]):
            lbl, h = get_label(k, train_sub_x, train_sub_y, valid_x[i])
            num_correct += (valid_y[i] == lbl)
            selected.append(h)
            labels.append(lbl)
        accuracy.append(num_correct / valid_x.shape[0])

        # Write the selected neighbors, predicted label, and actual labels
        with open(TABLES_PATH / ("p01_c_best_neighbors_k=%d.tex" % k), "w+") as f_out:
            for i, (lbl, neigh) in enumerate(zip(labels, selected)):
                id_num = num_train + i + 1
                if i > 0: f_out.write("\n")
                f_out.write("%d & " % id_num)
                neigh = sorted([x for _, x in neigh])
                for j, x in enumerate(neigh):
                    if j > 0: f_out.write(", ")
                    f_out.write("%d" % (x + 1))
                f_out.write("& %s" % lbl)
                f_out.write(r"& %s\\\hline" % train_y[num_train + i])

    # Find the k with the maximum accuracy so it can be highlighted in the tables
    best_k, best_acc = None, -np.inf
    for i, acc in enumerate(accuracy):
        if acc > best_acc:
            best_k, best_acc = i + 1, acc
    # Construct the best prediction table
    with open(TABLES_PATH / "p01_c_best_k.tex", "w+") as f_out:
        for i, acc in enumerate(accuracy):
            if i > 0:
                f_out.write("\n")
            if i == best_k - 1:
                start_str, end_str = r"\textbf{", r"}"
            else:
                start_str, end_str = r"", r""
            f_out.write(r"%d & %s%.3f%s \\\hline" % (i + 1, start_str, acc, end_str))
    return best_k


def predict_unlabel_with_k(k: int, train_x: torch.Tensor, train_y: torch.Tensor,
                           unlabel_x: torch.Tensor):
    r""" Generates the predictions for the unlabeled set with the best K """
    labels, selected = [], []
    for i in range(0, unlabel_x.shape[0]):
        lbl, h = get_label(k, train_x, train_y, unlabel_x[i])
        selected.append(h)
        labels.append(lbl)

    # Write the selected neighbors, predicted label, and actual labels
    with open(TABLES_PATH / "p01_c_best_neighbors_unlabel.tex", "w+") as f_out:
        for i, (lbl, neigh) in enumerate(zip(labels, selected)):
            if i > 0: f_out.write("\n")
            f_out.write("P%d & " % (i + 1))
            neigh = sorted([x for _, x in neigh])
            for j, x in enumerate(neigh):
                if j > 0: f_out.write(", ")
                f_out.write("%d" % (x + 1))
            f_out.write(r"& %s \\\hline" % lbl)


def _main(train: Path, unlabel: Path):
    train_x, train_y, transform_fields = extract_dataset(train, add_id_col=True)
    unlabel_x, _, _ = extract_dataset(unlabel, transform_fields)

    predict_part_b(train_x, train_y, unlabel_x, TABLES_PATH / "p01_part_b_")

    best_k = get_best_k(6, train_x, train_y)
    print("Best K = %d" % best_k)
    predict_unlabel_with_k(best_k, train_x, train_y, unlabel_x)


if __name__ == "__main__":
    _main(Path(sys.argv[1]), Path(sys.argv[2]))
