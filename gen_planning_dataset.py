import os
import gzip
import argparse
import pickle
from planning_dataset import ListSortingEnv, GraphPathEnv

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['sort', 'sort-15', 'shortest-path', 'shortest-path-25', 'shortest-path-10', 'shortest-path-15'])
parser.add_argument('--size', type=int, default=100000)
FLAGS = parser.parse_args()


def main():
    base_dir = './data/planning'
    os.makedirs(base_dir, exist_ok=True)

    if FLAGS.dataset == 'sort':
        env = ListSortingEnv(10)
        ds = env.generate_data(FLAGS.size)
    elif FLAGS.dataset == 'sort-15':
        env = ListSortingEnv(15)
        ds = env.generate_data(FLAGS.size)
    elif FLAGS.dataset == 'shortest-path':
        env = GraphPathEnv(20, (4, 5))
        ds = env.generate_data(FLAGS.size)
    elif FLAGS.dataset == 'shortest-path-10':
        env = GraphPathEnv(10, (2, 4))
        ds = env.generate_data(FLAGS.size)
    elif FLAGS.dataset == 'shortest-path-15':
        env = GraphPathEnv(15, (2, 4))
        ds = env.generate_data(FLAGS.size)
    elif FLAGS.dataset == 'shortest-path-25':
        env = GraphPathEnv(25, (4, 5))
        ds = env.generate_data(FLAGS.size)
    else:
        raise ValueError(f'Unknown dataset: {FLAGS.dataset}.')

    filename = os.path.join(base_dir, f'{FLAGS.dataset}-{FLAGS.size}.pkl.gz')
    with gzip.open(filename, 'wb') as f:
        pickle.dump(ds, f)
    print('Dataset saved to {}.'.format(filename))


if __name__ == '__main__':
    main()

