import networkx as nx
import matplotlib.pyplot as plt
import os

def read_file(file_path):
    with open(file_path) as file:
        return file.read()

def read_data_file(data_filename):
        return read_file(os.path.join('data', data_filename))
        
def get_test_data(name):
    dir_name = 'data'

    xml1_name = name + '_1.xml'
    xml2_name = name + '_2.xml'
    mapping_name = name + '_truth.txt'

    full_path = lambda n: os.path.join(dir_name, n)
    return list(map(read_file, map(full_path, [xml1_name, xml2_name, mapping_name])))

def display_graph(G):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
#     edge_labels = { (u, v): d['type'] for (u, v, d) in G.edges(data=True)}
    edge_labels = nx.get_edge_attributes(G, name='type')
    nx.draw_networkx(G)
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

def display_pairsiwe_graph(G):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, name='weight')
    node_labels = nx.get_node_attributes(G, name='score')
    nx.draw_networkx(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    nx.draw_networkx_labels(G, pos, node_labels)