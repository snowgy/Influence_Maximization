class Graph():
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
        self.network[s][e] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        return self.network[source].items()


