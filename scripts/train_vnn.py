#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vnn_release.data_utils import load_vnn_data, load_smiles_data  # noqa: E402
from vnn_release.training import (  # noqa: E402
    set_seed,
    run_all_id_models,
    run_fc_dnn_fold,
    run_lr_fold,
    run_polymoa_fold,
    run_rand_vnn_fold,
    run_xgb_fold,
    run_all_ood_models,
    run_polymoa_ood,
    run_fc_dnn_ood,
    run_rand_vnn_ood,
    run_xgb_ood,
    run_lr_ood,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models (ID CV) for T2D benchmark.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds to run.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["polymoa", "fc", "rand", "xgb", "lr"],
        default=["polymoa", "fc", "rand", "xgb", "lr"],
        help="Model set to run.",
    )
    parser.add_argument("--depth", type=int, default=4, help="Reactome hierarchy depth.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "id_metrics.csv",
        help="Where to write the ID CV metrics CSV.",
    )
    parser.add_argument(
        "--ood-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ood_metrics.csv",
        help="Where to write the OOD metrics CSV.",
    )
    parser.add_argument("--run-ood", action="store_true", help="Run OOD after ID.")
    parser.add_argument("--ood-only", action="store_true", help="Run only OOD (skip ID).")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = PROJECT_ROOT / "data"

    vnn_data = load_vnn_data(data_root / "raw", data_root / "mapping")
    pos_smiles_df, neg_smiles_df, val_smiles_df = load_smiles_data(data_root / "smiles")
    fold_splits = pickle.load(open(data_root / "splits" / "fold_splits.pkl", "rb"))
    ood_split = pickle.load(open(data_root / "splits" / "ood_split.pkl", "rb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Models: {args.models}")
    print(f"Run OOD: {args.run_ood or args.ood_only}")
    print(f"OOD only: {args.ood_only}")

    records = []

    # Run ID cross-validation unless skipping
    if not args.ood_only:
        if set(args.models) == {"polymoa", "fc", "rand", "xgb", "lr"}:
            print("Running all ID models...")
            records = run_all_id_models(
                vnn_data,
                pos_smiles_df,
                neg_smiles_df,
                fold_splits,
                seeds=args.seeds,
                builder_depth=args.depth,
                device=device,
            )
        else:
            for seed in args.seeds:
                print(f"ID: seed={seed}")
                set_seed(seed)
                for fold_idx in sorted(fold_splits.keys()):
                    print(f"  fold={fold_idx}")
                    if "polymoa" in args.models:
                        print("    model=polymoa")
                        m = run_polymoa_fold(vnn_data, fold_splits, fold_idx, builder_depth=args.depth, device=device)
                        records.append({"seed": seed, "fold": fold_idx, "model": "Polymoa-VNN", **m})
                        print(f"    -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "fc" in args.models:
                        print("    model=fc")
                        m = run_fc_dnn_fold(vnn_data, fold_splits, fold_idx, builder_depth=args.depth, device=device)
                        records.append({"seed": seed, "fold": fold_idx, "model": "FC-DNN", **m})
                        print(f"    -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "rand" in args.models:
                        print("    model=rand")
                        m = run_rand_vnn_fold(vnn_data, fold_splits, fold_idx, builder_depth=args.depth, device=device)
                        records.append({"seed": seed, "fold": fold_idx, "model": "RandMasked-VNN", **m})
                        print(f"    -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "xgb" in args.models:
                        print("    model=xgb")
                        m = run_xgb_fold(pos_smiles_df, neg_smiles_df, fold_splits, fold_idx, seed=seed)
                        records.append({"seed": seed, "fold": fold_idx, "model": "XGB", **m})
                        print(f"    -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "lr" in args.models:
                        print("    model=lr")
                        m = run_lr_fold(pos_smiles_df, neg_smiles_df, fold_splits, fold_idx, seed=seed)
                        records.append({"seed": seed, "fold": fold_idx, "model": "LR", **m})
                        print(f"    -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")

    if not args.ood_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.output, index=False)
        print(f"Saved metrics to {args.output}")
        if records:
            mean_aupr = sum(r["aupr"] for r in records) / len(records)
            mean_auroc = sum(r["auroc"] for r in records) / len(records)
            print(f"Mean AUROC={mean_auroc:.3f}, AUPR={mean_aupr:.3f}")

    if args.run_ood or args.ood_only:
        if set(args.models) == {"polymoa", "fc", "rand", "xgb", "lr"}:
            print("Running all OOD models...")
            ood_records = run_all_ood_models(
                vnn_data,
                pos_smiles_df,
                neg_smiles_df,
                ood_split,
                seeds=args.seeds,
                builder_depth=args.depth,
                device=device,
            )
        else:
            ood_records = []
            for seed in args.seeds:
                print(f"OOD: seed={seed}")
                set_seed(seed)
                for _ in [seed]:  # dummy loop for symmetry
                    if "polymoa" in args.models:
                        print("  model=polymoa")
                        m = run_polymoa_ood(vnn_data, ood_split, builder_depth=args.depth, device=device)
                        ood_records.append({"seed": seed, "model": "Polymoa-VNN", **m})
                        print(f"  -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "fc" in args.models:
                        print("  model=fc")
                        m = run_fc_dnn_ood(vnn_data, ood_split, builder_depth=args.depth, device=device)
                        ood_records.append({"seed": seed, "model": "FC-DNN", **m})
                        print(f"  -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "rand" in args.models:
                        print("  model=rand")
                        m = run_rand_vnn_ood(vnn_data, ood_split, builder_depth=args.depth, device=device)
                        ood_records.append({"seed": seed, "model": "RandMasked-VNN", **m})
                        print(f"  -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "xgb" in args.models:
                        print("  model=xgb")
                        m = run_xgb_ood(pos_smiles_df, neg_smiles_df, ood_split, seed=seed)
                        ood_records.append({"seed": seed, "model": "XGB", **m})
                        print(f"  -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
                    if "lr" in args.models:
                        print("  model=lr")
                        m = run_lr_ood(pos_smiles_df, neg_smiles_df, ood_split, seed=seed)
                        ood_records.append({"seed": seed, "model": "LR", **m})
                        print(f"  -> AUROC={m['auroc']:.3f} AUPR={m['aupr']:.3f}")
        args.ood_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ood_records).to_csv(args.ood_output, index=False)
        print(f"Saved OOD metrics to {args.ood_output}")
        if ood_records:
            mean_aupr = sum(r["aupr"] for r in ood_records) / len(ood_records)
            mean_auroc = sum(r["auroc"] for r in ood_records) / len(ood_records)
            print(f"OOD Mean AUROC={mean_auroc:.3f}, AUPR={mean_aupr:.3f}")


if __name__ == "__main__":
    main()
