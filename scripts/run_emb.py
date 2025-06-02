import sys
sys.path.append('./src')

import argparse
import time
from data import PPCI
from utils import get_time

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/v2", help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_proc", type=int, default=2, help="Number of processes")
    parser.add_argument("--environment", type=str, default="supervised", help="Environment")
    parser.add_argument("--generate", type=bool, default=True, help="Generate the dataset")
    parser.add_argument("--reduce_fps_factor", type=int, default=15, help="Reduce fps factor")
    parser.add_argument("--downscale_factor", type=float, default=1, help="Downscale factor")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--background", type=bool, default=False, help="Use background: True, Remove background: False")
    return parser

def main(args):
    encoders = ["dino"]#, "vit", "clip", "vit_large", "clip_large", "mae"]
    for encoder in encoders:
        PPCI(encoder = encoder,
             token = "class",
             task = "all",
             split_criteria = "all",
             environment = args.environment,
             batch_size = args.batch_size, 
             num_proc = args.num_proc,
             generate = args.generate, 
             data_dir = args.data_dir,
             verbose = args.verbose,
             background = args.background,)

if __name__ == "__main__":
    args = get_parser().parse_args()
    start_time = time.time()
    main(args)
    get_time(start_time)
