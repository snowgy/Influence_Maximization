from invgraph import Graph
from graph import pGraph
import random
import multiprocessing as mp
import time
import getopt
import sys
import math
import ISE


class Worker(mp.Process):
    def __init__(self, inQ, outQ):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0

    def run(self):

        while True:
            theta = self.inQ.get()
            # print(theta)
            while self.count < theta:
                v = random.randint(1, node_num)
                rr = generate_rr(v)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def create_worker(num):
    """
        create processes
        :param num: process number
        :param task_num: the number of tasks assigned to each worker
    """
    global worker
    for i in range(num):
        # print(i)
        worker.append(Worker(mp.Queue(), mp.Queue()))
        worker[i].start()


def finish_worker():
    """
    关闭所有子进程
    :return:
    """
    for w in worker:
        w.terminate()


def sampling(epsoid, l):
    global graph, seed_size, worker
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = 2
    create_worker(worker_num)
    for i in range(1, int(math.log2(n-1))+1):
        s = time.time()
        x = n/(math.pow(2, i))
        lambda_p = ((2+2*epsoid_p/3)*(logcnk(n, k) + l*math.log(n) + math.log(math.log2(n)))*n)/pow(epsoid_p, 2)
        theta = lambda_p/x
        # print(theta-len(R))
        for ii in range(worker_num):
            worker[ii].inQ.put((theta-len(R))/worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        # finish_worker()
        # worker = []
        end = time.time()
        # print('time to find rr', end - s)
        start = time.time()
        Si, f = node_selection(R, k)
        # print(f)
        end = time.time()
        # print('node selection time', time.time() - start)
        # print(F(R, Si))
        # f = F(R,Si)
        if n*f >= (1+epsoid_p)*x:
            LB = n*f/(1+epsoid_p)
            break
    # finish_worker()
    alpha = math.sqrt(l*math.log(n) + math.log(2))
    beta = math.sqrt((1-1/math.e)*(logcnk(n, k)+l*math.log(n)+math.log(2)))
    lambda_aster = 2*n*pow(((1-1/math.e)*alpha + beta), 2)*pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    # print(diff)
    _start = time.time()
    if diff > 0:
        # print('j')
        for ii in range(worker_num):
            worker[ii].inQ.put(diff/ worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''
    
    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    # print(_end - _start)
    finish_worker()
    return R


def generate_rr(v):
    global model
    if model == 'IC':
        return generate_rr_ic(v)
    elif model == 'LT':
        return generate_rr_lt(v)


def node_selection(R, k):
    Sk = set()
    rr_degree = [0 for ii in range(node_num+1)]
    node_rr_set = dict()
    node_rr_set_copy = dict()
    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            # print(rr_node)
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
                node_rr_set_copy[rr_node] = list()
            node_rr_set[rr_node].append(j)
            node_rr_set_copy[rr_node].append(j)

    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.add(max_point)
        matched_count += len(node_rr_set_copy[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    return Sk, matched_count/len(R)


def F(R, Si):
    matched_count = 0
    for rr in R:
        for s in Si:
            if s in rr:
                matched_count += 1
    return matched_count/len(R)

'''
def generate_rr_ic(node):
    # calculate reverse reachable set using IC model
    activity_set = set()
    activity_set.add(node)
    active_nodes = set()
    active_nodes.add(node)
    converged = False
    while not converged:
        new_activity_set = set()
        for seed in activity_set:
            for _node in set(graph.get_neighbors_keys(seed)) - active_nodes:
                weight = graph.network[seed][_node]
                if random.random() < weight:
                    active_nodes.add(_node)
                    new_activity_set.add(_node)
        activity_set = new_activity_set
        if not activity_set:
            converged = True
        active_nodes |= activity_set
    return active_nodes
'''


def generate_rr_ic(node):
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(node):
    # calculate reverse reachable set using LT model
    activity_set = list()
    activity_nodes = list()
    activity_set.append(node)
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            neighbors = graph.get_neighbors(seed)
            if len(neighbors) == 0:
                continue
            candidate = random.sample(neighbors, 1)[0][0]
            # print(candidate)
            if candidate not in activity_nodes:
                activity_nodes.append(candidate)
                new_activity_set.append(candidate)
        activity_set = new_activity_set
    return activity_nodes


def imm(epsoid, l):
    n = node_num
    k = seed_size
    l = l*(1+ math.log(2)/math.log(n))
    R = sampling(epsoid, l)
    Sk, z = node_selection(R, k)
    return Sk


def logcnk(n, k):
    res = 0
    for i in range(n-k+1, n+1):
        res += math.log(i)
    for i in range(1, k+1):
        res -= math.log(i)
    return res


def read_file(network):
    """
    read network file into a graph and read seed file into a list
    :param network: the file path of network
    """
    global node_num, edge_num, graph, seeds, pGraph
    data_lines = open(network, 'r').readlines()
    node_num = int(data_lines[0].split()[0])
    edge_num = int(data_lines[0].split()[1])

    for data_line in data_lines[1:]:
        start, end, weight = data_line.split()
        graph.add_edge(int(start), int(end), float(weight))
        pGraph.add_edge(int(start), int(end), float(weight))


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
    graph = Graph()
    pGraph = pGraph()
    model = 'IC'
    """
    command line parameters
    """
    start = time.time()
    network_path = "test_data/NetHEPT.txt"
    model = 'IC'
    seed_size = 0
    termination = 10
    # start = time.time()
    opts, args = getopt.getopt(sys.argv[1:], 'i:k:m:t:')
    for opt, val in opts:
        if opt == '-i':
            network_path = val
        elif opt == '-k':
            seed_size = int(val)
        elif opt == '-m':
            model = val
        elif opt == '-t':
            termination = int(val)

    read_file(network_path)
    worker = []
    epsoid = 0.1
    l = 1
    seeds = imm(epsoid, l)
    # print(seeds)

    for seed in seeds:
        print(seed)

    end = time.time()
    # print(end - start)
    #
    # res = ISE.calculate_influence(seeds, model, pGraph)
    #
    # print(res)


