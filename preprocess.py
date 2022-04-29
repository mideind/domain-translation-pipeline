from collections import Counter, defaultdict
from pathlib import Path
import sys
import fileinput
import random

# argv1 is a path to a .tsv file (tab separated value file) with the following format (without braces):
#     {space separated list of domains}\t{english segment}\t{icelandic segment}
# argv2 is a path to a directory which will contain the outputs
# 
# this program merely filters and removes domains that occur too infrequently and separates the tsv file into its constituent columns
#
# Usage:  python preprocess.py myfile.tsv ./myoutputdir

examples = []
counter = Counter()
input_path = Path(sys.argv[1])
assert input_path.exists()
output_dir = Path(sys.argv[2])
output_dir.mkdir(exist_ok=True)

with fileinput.input(files=[input_path]) as in_fh:
    for line in in_fh:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        domains, pair = line.split("\t", 1)
        if domains:
            counter.update(domains.split(" "))
        examples.append((domains, pair))

# if only using the 'single_domain.ge_k_toks.tsv' file, the least common one has 1579 pairs with ndomains=40
num_domains = 10000
min_domain_occur = 0
common_domains = [domain for  domain, count in  counter.items() if count >= min_domain_occur]
print("Using domains:", common_domains)

with (output_dir / "domains.txt").open("w") as fh_out:
    for domain in common_domains:
        fh_out.write(f"{domain} {counter[domain]}" + "\n")

random.seed(1)

with (output_dir / "train.domains").open("w") as dom_train_out, \
    (output_dir / "train.en").open("w") as en_train_out, \
    (output_dir / "train.is").open("w") as is_train_out:
    pairs = examples
    random.shuffle(pairs)
    ntrain = len(pairs)

    for subset, dom_fh, en_fh, is_fh in [
        (pairs[:ntrain], dom_train_out, en_train_out, is_train_out), 
    ]:
        dom_fh.write("\n".join([pair[0] for pair in subset]))
        en_fh.write("\n".join([pair[1].split("\t")[0] for pair in subset]))
        is_fh.write("\n".join([pair[1].split("\t")[1] for pair in subset]))
