from typing import Optional

import random
import numpy as np
import gzip
import pickle
from tqdm.auto import tqdm
from tabulate import tabulate
from reasoning_dataset import random_generate_graph, random_generate_graph_dnc, random_generate_special_graph
from torch.utils.data.dataset import Dataset


class PlanningDataset(Dataset):
    def __init__(self, dataset_identifier: str, split: str, num_identifier='100000'):
        self._dataset_identifier = dataset_identifier
        self._split = split
        self._num_identifier = num_identifier

        if self._dataset_identifier == 'sort':
            self._init_load()
        elif self._dataset_identifier == 'sort-15':
            self._init_load()
        elif self._dataset_identifier == 'shortest-path':
            self._init_load()
        elif self._dataset_identifier == 'shortest-path-1d':
            self._init_load_1d()
        elif self._dataset_identifier == 'shortest-path-10-1d':
            self._init_load_1d()
        elif self._dataset_identifier == 'shortest-path-15-1d':
            self._init_load_1d()
        elif self._dataset_identifier == 'shortest-path-25-1d':
            self._init_load_1d()
        else:
            raise ValueError('Unknown dataset identifier: {}.'.format(self._dataset_identifier))

        self.inp_dim = self._all_condition[0].shape[-1]
        self.out_dim = self._all_output[0].shape[-1]

    @classmethod
    def _load_data_raw(cls, identifier, num_identifier):
        if hasattr(cls, f'_data_{identifier}_{num_identifier}_raw'):
            return getattr(cls, f'_data_{identifier}_{num_identifier}_raw')

        with gzip.open('data/planning/{}-{}.pkl.gz'.format(identifier, num_identifier), 'rb') as f:
            all_data = pickle.load(f)

        setattr(cls, f'_data_{identifier}_{num_identifier}_raw', all_data)
        return all_data

    def _init_load(self):
        print('Loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))

        all_data = self._load_data_raw(self._dataset_identifier, self._num_identifier)
        if self._split == 'train':
            all_data = all_data[:int(0.9 * len(all_data))]
        elif self._split == 'validation':
            all_data = all_data[int(0.9 * len(all_data)):int(1.0 * len(all_data))]
        else:
            raise ValueError('Unknown split: {}.'.format(self._split))

        padding = 16
        all_condition, all_output = list(), list()
        for data in tqdm(all_data, desc='Preprocessing the data'):
            states = data['states']
            actions = data['actions']

            if self._dataset_identifier == 'shortest-path':
                initial_state = states[0][:, 0, -1].argmax()
                actions = [initial_state] + actions
                actions = [(x, y) for x, y in zip(actions[:-1], actions[1:])]

            n = states[0].shape[0]

            states_concat = np.stack(states + [states[-1] for _ in range(padding - len(states))], axis=0)
            actions = np.array(actions + [(0, 0) for _ in range(padding - len(actions))], dtype='int32')
            actions_onehot = np.zeros((padding, n, n, 1), dtype=np.float32)
            actions_onehot[np.arange(padding), actions[:, 0], actions[:, 1], 0] = 1

            condition = data['states'][0]
            output = np.concatenate([states_concat, actions_onehot], axis=-1)
            all_condition.append(condition)
            all_output.append(output)

        self._all_condition = np.stack(all_condition, axis=0)
        self._all_output = np.stack(all_output, axis=0)

        # normalize to -1 to 1
        self._all_condition = (self._all_condition - 0.5) * 2
        self._all_output = (self._all_output - 0.5) * 2

        print('Finished loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))

    def _init_load_1d(self):
        print('Loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))

        if self._dataset_identifier.startswith('shortest-path') and self._dataset_identifier.endswith('1d'):
            pass
        else:
            raise NotImplementedError('1D inputs are only supported for shortest-path.')

        # remove the -1d suffix
        all_data = self._load_data_raw(self._dataset_identifier[:-3], self._num_identifier)
        if self._split == 'train':
            all_data = all_data[:int(0.9 * len(all_data))]
        elif self._split == 'validation':
            all_data = all_data[int(0.9 * len(all_data)):int(1.0 * len(all_data))]
        else:
            raise ValueError('Unknown split: {}.'.format(self._split))

        padding = 8
        all_condition, all_output = list(), list()
        for data in tqdm(all_data, desc='Preprocessing the data'):
            states = data['states']
            actions = data['actions']

            n = states[0].shape[0]
            actions = actions + [actions[-1] for _ in range(padding - len(actions))]
            actions = np.array(actions, dtype='int32')
            actions_onehot = np.zeros((padding, n, 1), dtype=np.float32)
            actions_onehot[np.arange(padding), actions, 0] = 1

            condition = np.concatenate([states[0], states[-1]], axis=-1)
            output = actions_onehot
            all_condition.append(condition)
            all_output.append(output)

            # print(condition[:, 0, 1].argmax(), condition[:, 0, 3].argmax(), actions)

        self._all_condition = np.stack(all_condition, axis=0)
        self._all_output = np.stack(all_output, axis=0)

        # normalize to -1 to 1
        self._all_condition = (self._all_condition - 0.5) * 2
        self._all_output = (self._all_output - 0.5) * 2

        print('Finished loading dataset {}-{}...'.format(self._dataset_identifier, self._num_identifier))

    def __len__(self):
        return self._all_condition.shape[0]

    def __getitem__(self, index):
        return self._all_condition[index], self._all_output[index]


class PlanningDatasetOnline(object):
    def __init__(self, inner_env, n: Optional[int] = None):
        if isinstance(inner_env, str):
            if inner_env == 'list-sorting-2':
                assert n is not None
                inner_env = ListSortingEnv2(n)
            else:
                raise ValueError('Unknown inner env: {}.'.format(inner_env))

        self._inner_env = inner_env
        self._inner_env.reset()

        if isinstance(self.inner_env, ListSortingEnv2):
            self.dataset_mode = 'list-sorting-2'
        else:
            raise ValueError('Unknown inner env: {}.'.format(self.inner_env))

        self.inp_dim = 1
        self.out_dim = 3

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        if self.dataset_mode == 'list-sorting-2':
            return self._get_item_list_sorting_2(index)

    @property
    def inner_env(self):
        return self._inner_env

    def _get_item_list_sorting_2(self, index):
        obs = self.inner_env.reset()
        states, actions = [obs], list()
        while True:
            action = self.inner_env.oracle_policy(obs)
            if action is None:
                raise RuntimeError('No action found.')
            obs, _, finished, _ = self.inner_env.step(action)
            states.append(obs)
            actions.append(action)

            if finished:
                break

        padding = 16
        states = states + [states[-1] for _ in range(padding - len(states))]
        states = np.stack(states, axis=0)[:, :, np.newaxis]
        actions = actions + [(0, 0) for _ in range(padding - len(actions))]
        actions_onehot = np.zeros((states.shape[0], states.shape[1], 2), dtype=np.float32) - 1  # Instead of [0, 1], we use [-1, 1] 1]
        actions_onehot[np.arange(states.shape[0]), [a[0] for a in actions], 0] = 1
        actions_onehot[np.arange(states.shape[0]), [a[1] for a in actions], 1] = 1

        condition = states[0]
        output = np.concatenate([states, actions_onehot], axis=-1)
        return condition, output


class ListSortingEnv(object):
    """Env for sorting a random permutation."""

    def __init__(self, nr_numbers, np_random=None):
        super().__init__()
        self._nr_numbers = nr_numbers
        self._array = None
        self._np_random = np_random or np.random

    def reset_nr_numbers(self, n):
        self._nr_numbers = n
        self.reset()

    @property
    def array(self):
        return self._array

    @property
    def nr_numbers(self):
        return self._nr_numbers

    @property
    def np_random(self):
        return self._np_random

    def get_state(self):
        """ Compute the state given the array. """
        x, y = np.meshgrid(self.array, self.array)
        number_relations = np.stack([x < y, x == y, x > y], axis=-1).astype('float')
        index = np.array(list(range(self._nr_numbers)))
        x, y = np.meshgrid(index, index)
        position_relations = np.stack([x < y, x == y, x > y], axis=-1).astype('float')
        return np.concatenate([number_relations, position_relations], axis=-1)

    def _calculate_optimal(self):
        """ Calculate the optimal number of steps for sorting the array. """
        a = self._array
        b = [0 for i in range(len(a))]
        cnt = 0
        for i, x in enumerate(a):
            if b[i] == 0:
                j = x
                b[i] = 1
                while b[j] == 0:
                    b[j] = 1
                    j = a[j]
                assert i == j
                cnt += 1
        return len(a) - cnt

    def reset(self):
        """ Restart: Generate a random permutation. """
        while True:
            self._array = self.np_random.permutation(self._nr_numbers)
            self.optimal = self._calculate_optimal()
            if self.optimal > 0:
                break
        return self.get_state()

    def step(self, action):
        """
            Action: Swap the numbers at the index :math:`i` and :math:`j`.
            Returns: reward, is_over
        """
        a = self._array
        i, j = action
        x, y = a[i], a[j]
        a[i], a[j] = y, x
        for i in range(self._nr_numbers):
            if a[i] != i:
                return self.get_state(), 0, False, {}
        return self.get_state(), 1, True, {}

    def oracle_policy(self, state):
        """ Oracle policy: Swap the first two numbers that are not sorted. """
        a = self._array
        for i in range(self._nr_numbers):
            if a[i] != i:
                for j in range(i + 1, self._nr_numbers):
                    if a[j] == i:
                        return i, j
        return None

    def generate_data(self, nr_data_points: int):
        data = list()
        for _ in tqdm(range(nr_data_points)):
            obs = self.reset()
            states, actions = [obs], list()
            while True:
                action = self.oracle_policy(obs)
                if action is None:
                    raise RuntimeError('No action found.')
                obs, _, finished, _ = self.step(action)
                states.append(obs)
                actions.append(action)

                if finished:
                    break
            data.append({'states': states, 'actions': actions, 'optimal_steps': self.optimal, 'actual_steps': len(actions)})
        return data


class ListSortingEnv2(ListSortingEnv):
    """Env for sorting a random permutation. In constrast to :class:`ListSortingEnv`, this env uses a linear (instead of relational) state representation. Furthermore, the actions are represented as two one-hot vectors."""

    def get_state(self):
        """Return the raw array basically."""
        return (np.array(self.array / self.nr_numbers) - 0.5) * 2


class GraphEnvBase(object):
    """Graph Env Base."""

    def __init__(self, nr_nodes, p=0.5, directed=True, gen_method='edge'):
        """Initialize the graph env.

        Args:
            n: The number of nodes in the graph.
            p: Parameter for random generation. (Default 0.5)
                (edge method): The probability that a edge doesn't exist in directed graph.
                (dnc method): Control the range of the sample of out-degree.
                other methods: Unused.
            directed: Directed or Undirected graph. Default: ``False``(undirected)
            gen_method: Use which method to randomly generate a graph.
                'edge': By sampling the existance of each edge.
                'dnc': Sample out-degree (:math:`m`) of each nodes, and link to nearest neighbors in the unit square.
                'list': generate a chain-like graph.
        """
        super().__init__()
        self._nr_nodes = nr_nodes
        self._p = p
        self._directed = directed
        self._gen_method = gen_method
        self._graph = None

    @property
    def graph(self):
        return self._graph

    def _gen_graph(self):
        """ generate the graph by specified method. """
        n = self._nr_nodes
        p = self._p
        if self._gen_method in ['edge', 'dnc']:
            gen = random_generate_graph if self._gen_method == 'edge' else random_generate_graph_dnc
            self._graph = gen(n, p, self._directed)
        else:
            self._graph = random_generate_special_graph(n, self._gen_method, self._directed)


class GraphPathEnv(GraphEnvBase):
    """Env for Finding a path from starting node to the destination."""

    def __init__(self, nr_nodes, dist_range, p=0.5, directed=True, gen_method='dnc'):
        super().__init__(nr_nodes, p, directed, gen_method)
        self._dist_range = dist_range

    @property
    def dist(self):
        return self._dist

    def reset(self):
        """Restart the environment."""
        self._dist = self._sample_dist()
        self._task = None
        while True:
            self._gen_graph()
            task = self._gen_task()
            if task is not None:
                break
        self._dist_matrix = task[0]
        self._task = (task[1], task[2])
        self._current = self._task[0]
        self._steps = 0
        return self.get_state()

    def _sample_dist(self):
        lower, upper = self._dist_range
        upper = min(upper, self._nr_nodes - 1)
        return random.randint(0, upper - lower + 1) + lower

    def _gen_task(self):
        """Sample the starting node and the destination according to the distance."""
        dist_matrix = self._graph.get_shortest()
        st, ed = np.where(dist_matrix == self.dist)
        if len(st) == 0:
            return None
        ind = random.randint(0, len(st) - 1)
        return dist_matrix, st[ind], ed[ind]

    def get_state(self):
        relation = self._graph.get_edges()
        current_state = np.zeros_like(relation)
        current_state[self._current, :] = 1
        return np.stack([relation, current_state], axis=-1)

    def step(self, action):
        """Move to the target node from current node if has_edge(current -> target)."""
        if self._current == self._task[1]:
            return self.get_state(), 1, True, {}
        if self._graph.has_edge(self._current, action):
            self._current = action
        else:
            return self.get_state(), -1, False, {}
        if self._current == self._task[1]:
            return self.get_state(), 1, True, {}
        self._steps += 1
        if self._steps >= self.dist:
            return self.get_state(), 0, True, {}
        return self.get_state(), 0, False, {}

    def oracle_policy(self, state):
        """Oracle policy: Swap the first two numbers that are not sorted."""
        current = self._current
        target = self._task[1]
        if current == target:
            return target
        possible_actions = state[current, :, 0] == 1
        # table = list()
        # table.append(('connected', possible_actions.nonzero()[0]))
        possible_actions = possible_actions & (self._dist_matrix[:, target] < self._dist_matrix[current, target])
        # table.append(('dist', possible_actions.nonzero()[0]))
        # print(tabulate(table, headers=['name', 'list']))
        if np.sum(possible_actions) == 0:
            raise RuntimeError('No action found.')
        return np.random.choice(np.where(possible_actions)[0])

    def generate_data(self, nr_data_points: int):
        data = list()
        for _ in tqdm(range(nr_data_points)):
            obs = self.reset()
            states, actions = [obs], list()
            while True:
                action = self.oracle_policy(obs)
                if action is None:
                    raise RuntimeError('No action found.')
                obs, reward, finished, _ = self.step(action)
                states.append(obs)
                actions.append(action)

                assert reward >= 0

                if finished:
                    break
            # import ipdb; ipdb.set_trace()
            data.append({'states': states, 'actions': actions, 'optimal_steps': self._dist, 'actual_steps': len(actions)})
        return data

