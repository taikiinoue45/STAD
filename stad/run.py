from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('yaml', help='yaml path', type=str)
args = parser.parse_args()
builder = CustomBuilder(args.yaml)
builder.run()