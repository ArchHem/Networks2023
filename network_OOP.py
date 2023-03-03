import numpy as np
from tqdm import tqdm
from random import choice
import matplotlib.pyplot as plt
from logbin_2020 import logbin

class Network:
    def __init__(self,init_state):
        self.state = init_state
        self.edge_count = int(sum( [ len(listElem) for listElem in self.state])/2)
        self.vertices_count = len(self.state)

    def get_edge_count(self):

        return int(sum( [ len(listElem) for listElem in self.state])/2)

    def get_vertice_count(self):
        return len(self.state)

    def append_vertex(self):
        self.state.append([])
        self.edge_count = self.get_edge_count()
        self.vertices_count = self.get_vertice_count()

    def edge_list_gen(self):

        edges_count = [len(length) for length in self.state]

        return edges_count

    def estabilish_connection(self,conn_id,connected_id=True):
        if connected_id:
            connected_id = len(self.state) - 1
        self.state[connected_id].append(conn_id)
        self.state[conn_id].append(connected_id)
        self.edge_count = self.get_edge_count()
        self.vertices_count = self.get_vertice_count()

    def lin_pref_connecter(self,m):
        self.append_vertex()
        total_edges_double = 2*self.get_edge_count()
        local_edge_count = self.edge_list_gen()[:-1]
        """
        indices = np.array([i for i in range(len(self.state))],dtype='int')

        crit_indices = np.where(np.array(local_edge_count)/total_edges_double > np.array(probs))

        scan_indeces = indices[crit_indices]
        """

        count = 0
        index_list = [j for j in range(len(self.state)-1)]

        while m > count:
            i = choice(index_list)
            if local_edge_count[i]/total_edges_double > np.random.uniform():
                self.estabilish_connection(i)
                count = count + 1
                index_list.remove(i)

    def lin_pref_time_evul(self,NT,m):
        for t in tqdm(range(NT)):
            self.lin_pref_connecter(m)


"""
model = Network([[1,2,3,4,5],[0,2,3,4,5],[0,1,3,4,5],[0,1,2,4,5],[0,1,2,3,5],[0,1,2,3,4]])
model.lin_pref_time_evul(10000,5)

ks = np.array(model.edge_list_gen())
x,y = logbin(ks,scale=1.2)
plt.plot(x,y)
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()"""







