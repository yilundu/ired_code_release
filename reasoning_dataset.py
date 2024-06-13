import copy
import numpy as np
import numpy.random as npr

from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST

__all__ = [
    'Graph', 'Family',
    'random_generate_graph', 'random_generate_graph_dnc', 'random_generate_special_graph', 'random_generate_family',
    'GraphOutDegreeDataset', 'GraphConnectivityDataset', 'GraphAdjacentDataset', 'FamilyTreeDataset'
]


class Graph(object):
    """Store a graph using adjacency matrix."""

    def __init__(self, nr_nodes, edges, coordinates=None):
        """Initialize a graph.

        Args:
            nr_nodes: The Number of nodes in the graph.
            edges: The adjacency matrix of the graph.
        """
        edges = edges.astype('int32')
        assert edges.min() >= 0 and edges.max() <= 1
        self._nr_nodes = nr_nodes
        self._edges = edges
        self._coordinates = coordinates
        self._shortest = None
        self.extra_info = {}

    def get_edges(self):
        return copy.copy(self._edges)

    def get_coordinates(self):
        return self._coordinates

    def get_relations(self):
        """ Return edges and identity matrix """
        return np.stack([self.get_edges(), np.eye(self._nr_nodes)], axis=-1)

    def has_edge(self, x, y):
        return self._edges[x, y] == 1

    def get_out_degree(self):
        """ Return the out degree of each node. """
        return np.sum(self._edges, axis=1)

    def get_shortest(self):
        """ Return the length of shortest path between nodes. """
        if self._shortest is not None:
            return self._shortest

        n = self._nr_nodes
        edges = self.get_edges()

        # n + 1 indicates unreachable.
        shortest = np.ones((n, n)) * (n + 1)
        shortest[np.where(edges == 1)] = 1
        # Make sure that shortest[x, x] = 0
        shortest -= shortest * np.eye(n)
        shortest = shortest.astype(np.int32)

        # Floyd Algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        shortest[i, j] = min(shortest[i, j], shortest[i, k] + shortest[k, j])
        self._shortest = shortest
        return self._shortest

    def get_connectivity(self, k=None, exclude_self=True):
        """Return the k-connectivity for each pair of nodes. It will return the full connectivity matrix if k is None or k < 0.
        When exclude_self is True, the diagonal elements will be 0.

        Args:
            k: the k-connectivity. Default: ``None``(full connectivity).
            exclude_self: exclude the diagonal elements. Default: ``True``.

        Returns:
            conn: The connectivity matrix.
        """
        shortest = self.get_shortest()
        if k is None or k < 0:
            k = self._nr_nodes
        k = min(k, self._nr_nodes)
        conn = (shortest <= k).astype(np.int32)
        if exclude_self:
            n = self._nr_nodes
            inds = np.where(~np.eye(n, dtype=np.bool_))
            conn = conn[inds]
            conn.resize(n, n - 1)
        return conn


def random_generate_special_graph(n: int, graph_type: str, directed: bool = False):
    """Randomly generate a special type graph.

    For list graph, the nodes are randomly permuted and connected in order. If the graph is directed, the edges are
    directed from the first node to the last node.

    Args:
        n: The number of nodes in the graph.
        graph_type: The type of the graph, e.g. list, tree. Currently only support list.
        directed: Directed or Undirected graph. Default: ``False``(undirected).

    Returns:
        graph: Generated graph.
    """
    if graph_type == 'list':
        nodes = npr.permutation(n)
        edges = np.zeros((n, n))
        for i in range(n - 1):
            x, y = nodes[i], nodes[i + 1]
            if directed:
                edges[x, y] = 1
            else:
                edges[x, y] = edges[y, x] = 1
        graph = Graph(n, edges)
        graph.extra_info['nodes_list'] = nodes
        return graph
    else:
        assert False, "not supported graph type: {}".format(graph_type)


def random_generate_graph(n, p, directed=False):
    """Randomly generate a graph by sampling the existence of each edge.
    Each edge between nodes has the probability :math:`p`(directed) or :math:`p^2`(undirected) to not exist.

    This paradigm is also called the Erdős–Rényi model.

    Args:
        n: The number of nodes in the graph.
        p: the probability that a edge doesn't exist in directed graph.
        directed: Directed or Undirected graph. Default: ``False``(undirected)

    Returns:
        graph: Generated graph.
    """
    edges = (npr.rand(n, n) < p).astype(np.float32)
    edges -= edges * np.eye(n)
    if not directed:
        edges = np.maximum(edges, edges.T)
    return Graph(n, edges)


def random_generate_graph_dnc(n, p=None, directed=False):
    """Random graph generation method as in DNC, the Differentiable Neural Computer paper.
    Sample :math:`n` nodes in a unit square. sample out-degree (:math:`m`) of each nodes,
    connect to :math:`m` nearest neighbors (Euclidean distance) in the unit square.

    Args:
        n: The number of nodes in the graph.
        p: Control the range of the sample of out-degree. Default: (1, n // 3)
            (float): (1, int(n * p))
            (int): (1, p)
            (tuple): (p[0], p[1])
        directed: Directed or Undirected graph. Default: ``False``(undirected)

    Returns:
        graph: A randomly generated graph.
    """
    edges = np.zeros((n, n), dtype=np.float32)
    pos = npr.rand(n, 2)

    def dist(x, y):
        return ((x - y) ** 2).mean()

    if type(p) is tuple:
        lower, upper = p
    else:
        lower = 1
        if p is None:
            upper = n // 3
        elif type(p) is int:
            upper = p
        elif type(p) is float:
            upper = int(n * p)
        else:
            assert False
        upper = max(upper, 1)
    lower = max(lower, 1)
    upper = min(upper, n - 1)

    for i in range(n):
        d = []
        k = npr.randint(upper - lower + 1) + lower
        for j in range(n):
            if i != j:
                d.append((dist(pos[i], pos[j]), j))
        d.sort()
        for j in range(k):
            edges[i, d[j][1]] = 1
    if not directed:
        edges = np.maximum(edges, edges.T)
    return Graph(n, edges, pos)


class Family(object):
    """A data structure that stores the relationship between N people in a family."""

    def __init__(self, nr_people: int, relations: np.ndarray):
        """Initialize a family with relations.

        Args:
            nr_people: number of people in the family.
            relations: a 3D array of shape (nr_people, nr_people, 6), where
                relations[i, j, 0] = 1 if j is the husband of i, 0 otherwise.
                relations[i, j, 1] = 1 if j is the wife of i, 0 otherwise.
                relations[i, j, 2] = 1 if j is the father of i, 0 otherwise.
                relations[i, j, 3] = 1 if j is the mother of i, 0 otherwise.
                relations[i, j, 4] = 1 if j is the son of i, 0 otherwise.
                relations[i, j, 5] = 1 if j is the daughter of i, 0 otherwise.
        """
        self._n = nr_people
        self._relations = relations

    @property
    def father(self) -> np.ndarray:
        return self._relations[:, :, 2]

    @property
    def mother(self) -> np.ndarray:
        return self._relations[:, :, 3]

    @property
    def son(self) -> np.ndarray:
        return self._relations[:, :, 4]

    @property
    def daughter(self) -> np.ndarray:
        return self._relations[:, :, 5]

    def has_father(self) -> np.ndarray:
        return self.father.max(axis=1)

    def has_daughter(self) -> np.ndarray:
        return self.daughter.max(axis=1)

    def has_sister(self) -> np.ndarray:
        return _clip_mul(self.father, self.daughter).max(axis=1)

    def get_parents(self) -> np.ndarray:
        return np.clip(self.father + self.mother, 0, 1)

    def get_grandfather(self) -> np.ndarray:
        return _clip_mul(self.get_parents(), self.father)

    def get_grandmother(self) -> np.ndarray:
        return _clip_mul(self.get_parents(), self.mother)

    def get_grandparents(self) -> np.ndarray:
        parents = self.get_parents()
        return _clip_mul(parents, parents)

    def get_uncle(self) -> np.ndarray:
        return _clip_mul(self.get_grandparents(), self.son)

    def get_maternal_great_uncle(self) -> np.ndarray:
        return _clip_mul(_clip_mul(self.get_grandmother(), self.mother), self.son)


def random_generate_family(n, p_marriage=0.8, verbose=False) -> Family:
    assert n > 0
    ids = list(npr.permutation(n))

    single_m = []
    single_w = []
    couples = [None]
    rel = np.zeros((n, n, 6))  # husband, wife, father, mother, son, daughter
    fathers = [None for i in range(n)]
    mothers = [None for i in range(n)]

    def add_couple(man, woman):
        couples.append((man, woman))
        rel[woman, man, 0] = 1  # husband
        rel[man, woman, 1] = 1  # wife
        if verbose:
            print('couple', man, woman)

    def add_child(parents, child, gender):
        father, mother = parents
        fathers[child] = father
        mothers[child] = mother
        rel[child, father, 2] = 1  # father
        rel[child, mother, 3] = 1  # mother
        if gender == 0:  # son
            rel[father, child, 4] = 1
            rel[mother, child, 4] = 1
        else:  # daughter
            rel[father, child, 5] = 1
            rel[mother, child, 5] = 1
        if verbose:
            print('child', father, mother, child, gender)

    def check_relations(man, woman):
        if fathers[man] is None or fathers[woman] is None:
            return True
        if fathers[man] == fathers[woman]:
            return False

        def same_parent(x, y):
            return fathers[x] is not None and fathers[y] is not None and fathers[x] == fathers[y]

        for x in [fathers[man], mothers[man]]:
            for y in [fathers[woman], mothers[woman]]:
                if same_parent(man, y) or same_parent(woman, x) or same_parent(x, y):
                    return False
        return True

    while len(ids) > 0:
        x = ids.pop()
        gender = npr.randint(2)
        parents = couples[npr.randint(len(couples))]
        if gender == 0:
            single_m.append(x)
        else:
            single_w.append(x)
        if parents is not None:
            add_child(parents, x, gender)

        if npr.rand() < p_marriage and len(single_m) > 0 and len(single_w) > 0:
            mi = npr.randint(len(single_m))
            wi = npr.randint(len(single_w))
            man = single_m[mi]
            woman = single_w[wi]
            if check_relations(man, woman):
                add_couple(man, woman)
                del single_m[mi]
                del single_w[wi]

    return Family(n, rel)


def _clip_mul(x, y):
    return np.clip(np.matmul(x, y), 0, 1)


class GraphDatasetBase(Dataset):
    def __init__(self, nr_nodes, p, epoch_size, directed=False, gen_method='dnc'):
        if type(nr_nodes) is int:
            self.nr_nodes = (max(nr_nodes // 2, 1), nr_nodes)
        else:
            self.nr_nodes = tuple(nr_nodes)
        self.p = p
        self.epoch_size = epoch_size
        self.directed = directed
        self.gen_method = gen_method
        assert self.gen_method in ('dnc', 'erdos_renyi')

    def _gen_graph(self, item):
        nr_nodes = item % (self.nr_nodes[1] - self.nr_nodes[0] + 1) + self.nr_nodes[0]
        if self.p is None:
            p = None
        elif type(self.p) is float:
            p = self.p
        else:
            p = self.p[0] + npr.rand() * (self.p[1] - self.p[0])
        gen_graph = random_generate_graph_dnc if self.gen_method == 'dnc' else random_generate_graph
        return gen_graph(nr_nodes, p, directed=self.directed)

    def __len__(self):
        return self.epoch_size


class GraphOutDegreeDataset(GraphDatasetBase):
    def __init__(self, nr_nodes, p, epoch_size, degree=2, directed=False, gen_method='dnc'):
        super().__init__(nr_nodes, p, epoch_size, directed, gen_method)
        self.degree = degree

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        return dict(
            n=graph._nr_nodes,
            relations=np.expand_dims(graph.get_edges(), axis=-1),
            target=(graph.get_out_degree() == self.degree).astype('float'),
        )


class GraphConnectivityDataset(GraphDatasetBase):
    def __init__(self, nr_nodes, p, epoch_size, dist_limit=None, directed=False, gen_method='dnc'):
        super().__init__(nr_nodes, p, epoch_size, directed, gen_method)
        self.dist_limit = dist_limit

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        return dict(
            n=graph._nr_nodes,
            relations=np.expand_dims(graph.get_edges(), axis=-1),
            # relations=graph.get_relations(),
            target=graph.get_connectivity(self.dist_limit, exclude_self=False),
        )


class GraphAdjacentDataset(GraphDatasetBase):
    def __init__(self, nr_nodes, p, epoch_size, nr_colors, directed=False, gen_method='dnc',
                 is_mnist_colors=False, is_train=True):

        super().__init__(nr_nodes, p, epoch_size, directed, gen_method)
        self._nr_colors = nr_colors
        self._mnist_colors = is_mnist_colors
        if is_mnist_colors:
            assert nr_colors == 10
            transform = None
            self.mnist = MNIST('../data', train=is_train, download=True, transform=transform)

    def __getitem__(self, item):
        graph = self._gen_graph(item)
        n = graph._nr_nodes
        if self._mnist_colors:
            m = self.mnist.__len__()
            digits = []
            colors = []
            for i in range(n):
                x = npr.randint(m)
                digit, color = self.mnist.__getitem__(x)
                digits.append(np.array(digit)[np.newaxis])
                colors.append(color)
            digits, colors = np.array(digits), np.array(colors)
        else:
            colors = npr.randint(self._nr_colors, size=n)
        states = np.zeros((n, self._nr_colors))
        adjacent = np.zeros((n, self._nr_colors))
        for i in range(n):
            states[i, colors[i]] = 1
            adjacent[i, colors[i]] = 1
            for j in range(n):
                if graph.has_edge(i, j):
                    adjacent[i, colors[j]] = 1
        if self._mnist_colors:
            states = digits
        return dict(
            n=n,
            relations=np.expand_dims(graph.get_edges(), axis=-1),
            states=states,
            colors=colors,
            target=adjacent,
            # connectivity=graph.get_connectivity(self.dist_limit, exclude_self=True),
        )


class FamilyTreeDataset(Dataset):
    def __init__(self, nr_people, epoch_size, task, p_marriage=0.8, balance_sample=False):
        super().__init__()
        if type(nr_people) is int:
            self.nr_people = (max(nr_people // 2, 1), nr_people)
        else:
            self.nr_people = tuple(nr_people)
        self.epoch_size = epoch_size
        self.task = task
        self.p_marriage = p_marriage
        self.balance_sample = balance_sample
        self.data = []

        assert task in ['has-father', 'has-daughter', 'has-sister', 'parents', 'grandparents', 'uncle', 'maternal-great-uncle']

    def _gen_family(self, item):
        nr_people = item % (self.nr_people[1] - self.nr_people[0] + 1) + self.nr_people[0]
        return random_generate_family(nr_people, self.p_marriage)

    def __getitem__(self, item):
        while len(self.data) == 0:
            family = self._gen_family(item)
            relations = family._relations[:, :, 2:]
            if self.task == 'has-father':
                target = family.has_father()
            elif self.task == 'has-daughter':
                target = family.has_daughter()
            elif self.task == 'has-sister':
                target = family.has_sister()
            elif self.task == 'parents':
                target = family.get_parents()
            elif self.task == 'grandparents':
                target = family.get_grandparents()
            elif self.task == 'uncle':
                target = family.get_uncle()
            elif self.task == 'maternal-great-uncle':
                target = family.get_maternal_great_uncle()
            else:
                assert False, "{} is not supported.".format(self.task)

            if not self.balance_sample:
                return dict(n=family._n, relations=relations, target=target)

            def get_position(x):
                return list(np.vstack(np.where(x)).T)

            def append_data(pos, target):
                states = np.zeros((family._n, 2))
                states[pos[0], 0] = states[pos[1], 1] = 1
                self.data.append(dict(n=family._n, relations=relations, states=states, target=target))

            positive = get_position(target == 1)
            if len(positive) == 0:
                continue
            negative = get_position(target == 0)
            npr.shuffle(negative)
            negative = negative[:len(positive)]
            for i in positive:
                append_data(i, 1)
            for i in negative:
                append_data(i, 0)

        return self.data.pop()

    def __len__(self):
        return self.epoch_size


class FamilyDatasetWrapper(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.inp_dim = 4
        self.out_dim = 1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        fd = self.ds[index]
        relations = fd['relations']
        target = fd['target']

        relations = (relations - 0.5) * 2
        target = (target - 0.5) * 2
        return relations, target


class GraphDatasetWrapper(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.inp_dim = 1
        self.out_dim = 1

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        fd = self.ds[index]
        relations = fd['relations']
        target = fd['target']

        relations = (relations - 0.5) * 2
        target = (target - 0.5) * 2
        return relations, target


