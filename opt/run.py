import random
import time
import numpy as np
import torch
from tracking import Tracker
from parser import get_parser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    setup_seed(20230606) 
    args = get_parser().parse_args()
    tracker = Tracker(args)

    t_start = time.perf_counter()

    tracker.run()

    t_end = time.perf_counter()

    print(f"Total runtime: {t_end-t_start:0.4f} seconds")

if __name__ == '__main__':
    main()

