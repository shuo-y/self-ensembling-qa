from model import BaselineReader
from model import EMA
from utils import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(sys.argv[0] + " path_to_gz_file")
        exit(0)

    prefix = sys.argv[1].split(".")[0]

    meta, samples = load_dataset(sys.argv[1])
    total = len(samples)
    print("Total %i data " % total)
    dev_part = int(total* 0.12)

    first = [meta] + samples[:dev_part]
    msg = "\n".join([json.dumps(s) for s in first])
    with gzip.open(prefix + "_dev.jsonl.gz", "w") as fout:
        fout.write(str.encode(msg))

    meta["header"]["split"] = "train"
    second = [meta] + samples[dev_part:]
    msg = "\n".join([json.dumps(s) for s in second])
    with gzip.open(prefix + "_train.jsonl.gz", "w") as fout:
        fout.write(str.encode(msg))
    print(meta)