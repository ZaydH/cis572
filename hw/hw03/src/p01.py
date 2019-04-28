import sys
from pathlib import Path

import torch


def extract_dataset(path: Path, mins = None, maxs = None, add_id_col = False):
    with open(path, "r") as f_in:
        lines = f_in.read().splitlines()
    spl = [l.split(",") for l in lines]
    labels = [l[-1] for l in spl]
    vals = torch.Tensor([[float(v.strip()) for v in l[:-1]] for l in spl])
    vals, mins, maxs = normalize(vals, mins, maxs)

    out_path = path.parent / (path.stem + "_normalized.tex")
    export_dataset(vals, labels, out_path, add_id_col)
    return vals, labels, mins, maxs


def export_dataset(x: torch.Tensor, y, out_file: Path, add_id_col):
    with open(out_file, "w+") as f_out:
        for r in range(x.shape[0]):
            if r > 0: f_out.write("\n")
            if add_id_col:
                f_out.write("%d & " % (r+1))
            for c in range(x.shape[1]):
                if c > 0: f_out.write(" & ")
                f_out.write("%.3f" % float(x[r, c]))
            f_out.write(r" & %s \\\hline" % y[r])


def normalize(vals: torch.Tensor, mins = None, maxs = None) -> torch.Tensor:
    if mins is None: mins = torch.min(vals, dim=0)[0]
    vals -= mins
    if maxs is None: maxs = torch.max(vals, dim=0)[0]
    return vals / maxs, mins, maxs


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
        row = unlabel_x[i]

        l1 = (train_x - row).abs()
        tot = l1.sum(dim=1)
        best = torch.argmin(tot)
        pred.append(train_y[best])

        path = Path(base_file_path + "p" + str(i+1) + "_err.tex")
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

    with open(base_file_path + "pred.tex", "w+") as f_out:
        for i, label in enumerate(pred):
            f_out.write(r"%d & %s \\\hline" % (i + 1, label))


def _main(train: Path, unlabel: Path):
    train_x, train_y, mins, maxs = extract_dataset(train, add_id_col=True)
    unlabel_x, _, _, _ = extract_dataset(unlabel, mins, maxs)

    predict_part_b(train_x, train_y, unlabel_x, "datasets/p01_part_b_")


if __name__ == "__main__":
    _main(Path(sys.argv[1]), Path(sys.argv[2]))
