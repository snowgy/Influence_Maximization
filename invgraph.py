class Graph:
    """
        graph data structure to store the network
    :return:
    """
    def __init__(self):
        self.network = dict()

    def add_node(self, node):
        if node not in self.network:
            self.network[node] = dict()

    def add_edge(self, s, e, w):
        """
        :param s: start node
        :param e: end node
        :param w: weight
        """
        Graph.add_node(self, s)
        Graph.add_node(self, e)
        # add inverse edge
        self.network[e][s] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        if source in self.network:
            return self.network[source].items()
        else:
            return []

    def get_neighbors_keys(self, source):
        if source in self.network:
            return self.network[source].keys()
        else:
            return []


