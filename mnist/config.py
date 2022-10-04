import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--batch_size', type=int, default=128, help='data batch size')
    parser.add_argument('--epochs', type=int, default=2, help='data batch size')

    return parser.parse_args()
