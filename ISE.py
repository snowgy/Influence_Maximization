from graph import pGraph
import random
import multiprocessing as mp
import time
import getopt
import sys


class Worker(mp.Process):
    def __init__(self, outQ, count):
        super(Worker, self).__init__(target=self.start)
        self.outQ = outQ
        self.count = count
        self.sum = 0

    def run(self):
        while self.count > 0:
            # print(self.count)
            res = ise()
            self.sum += res
            self.count -= 1
            if self.count == 0:
                self.outQ.put(self.sum)


def create_worker(num, task_num):
    """
        create processes
        :param num: process number
        :param task_num: the number of tasks assigned to each worker
    """
    for i in range(num):
        worker.append(Worker(mp.Queue(), task_num))
        worker[i].start()


def finish_worker():
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


def ise():
    global model
    if model == 'IC':
        return IC()
    elif model == 'LT':
        return LT()


def read_file(network, seed):
    """
    read network file into a graph and read seed file into a list
    :param network: the file path of network
    :param seed: the file path of seed
    """
    global node_num, edge_num, graph, seeds
    data_lines = open(network, 'r').readlines()
    seed_lines = open(seed, "r").readlines()
    node_num = int(data_lines[0].split()[0])
    edge_num = int(data_lines[0].split()[1])

    for data_line in data_lines[1: ]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), float(weight))

    for seed_line in seed_lines:
        seeds.append(int(seed_line))


def IC():
    """
    implement independent cascade model
    """
    global seeds, graph
    count = len(seeds)
    activity_set = set(seeds)
    active_nodes = set(seeds)
    while activity_set:
        new_activity_set = set()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in active_nodes:
                    if random.random() < weight:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def LT():
    """
    implement linear threshold model
    """
    global seeds, graph
    count = len(seeds)
    activity_set = set(seeds)
    active_nodes = set(seeds)
    node_threshold = {}
    node_weights = {}
    while activity_set:
        new_activity_set = set()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in active_nodes:
                    if node not in node_threshold:
                        node_threshold[node] = random.random()
                        node_weights[node] = 0
                    node_weights[node] += weight
                    if node_weights[node] >= node_threshold[node]:
                        active_nodes.add(node)
                        new_activity_set.add(node)
        count += len(new_activity_set)
        activity_set = new_activity_set
    return count


def calculate_influence(Sk, model_type, _graph):
    global seeds, worker, model, graph
    graph = _graph
    model = model_type
    seeds = Sk
    worker = []
    worker_num = 8
    create_worker(worker_num, int(10000 / worker_num))
    result = []
    for w in worker:
        # print(w.outQ.get())
        result.append(w.outQ.get())
    # print('%.2f' % (sum(result) / 10000))
    finish_worker()
    return sum(result) / 10000

if __name__ == "__main__":
    """
    define global variables:
    node_num: total number of nodes in the network
    edge_num: total number of edges in the network
    graph: represents the network
    seeds: the list of seeds
    """
    node_num = 0
    edge_num = 0
    graph = pGraph()
    seeds = []
    model = 'IC'
    """
    command line parameters
    """
    network_path = "test_data/NetHEPT.txt"
    seed_path = "test_data/seeds2.txt"
    model = 'IC'
    termination = 10
    start = time.time()
    opts, args = getopt.getopt(sys.argv[1:], 'i:s:m:t:')
    for opt, val in opts:
        if opt == '-i':
            network_path = val
        elif opt == '-s':
            seed_path = val
        elif opt == '-m':
            model = val
        elif opt == '-t':
            termination = val

    read_file(network_path, seed_path)

    worker = []
    worker_num = 2
    create_worker(worker_num, int(10000/worker_num))
    result = []
    for w in worker:
        # print(w.outQ.get())
        result.append(w.outQ.get())
    print('%.2f' % (sum(result) / 10000))
    finish_worker()
    end = time.time()
    print(end - start)
