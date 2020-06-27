import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('yaml', help='yaml path', type=str)
args = parser.parse_args()
with open(args.yaml) as f:
    Y = yaml.safe_load(f)