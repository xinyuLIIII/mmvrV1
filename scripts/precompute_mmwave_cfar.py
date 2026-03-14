import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.mmwave_cfar_dataset import precompute_mmwave_cfar_dataset


def build_parser():
    parser = argparse.ArgumentParser(description='Precompute offline CFAR mmwave dataset.')
    parser.add_argument('--dataset_root', default='./data')
    parser.add_argument('--cfar_mode', default='os2d', choices=('os2d',))
    parser.add_argument('--cfar_guard', default=1, type=int)
    parser.add_argument('--cfar_train', default=4, type=int)
    parser.add_argument('--cfar_rank_ratio', default=0.75, type=float)
    parser.add_argument('--cfar_pfa', default=1e-3, type=float)
    parser.add_argument('--cfar_soft_mode', default='subtract', choices=('subtract', 'mask'))
    parser.add_argument('--cfar_split_halves', dest='cfar_split_halves', action='store_true')
    parser.add_argument('--no_cfar_split_halves', dest='cfar_split_halves', action='store_false')
    parser.set_defaults(cfar_split_halves=True)
    return parser


def main():
    args = build_parser().parse_args()
    output_dir, file_count = precompute_mmwave_cfar_dataset(
        args.dataset_root,
        mode=args.cfar_mode,
        guard_cells=args.cfar_guard,
        training_cells=args.cfar_train,
        rank_ratio=args.cfar_rank_ratio,
        pfa=args.cfar_pfa,
        soft_mode=args.cfar_soft_mode,
        split_halves=args.cfar_split_halves,
    )
    print(f'precomputed {file_count} files into {output_dir}')


if __name__ == '__main__':
    main()
