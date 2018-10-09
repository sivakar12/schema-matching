import networkx as nx
import xml.etree.ElementTree as ET

def generate_graph(xml_string):
    def _recurse(Graph, node, parent_path='/'):
        node_name = parent_path + node.tag
        if node_name not in Graph.nodes:
            Graph.add_node(node_name)
        for attribute in node.attrib.keys():
            attribute_name = parent_path[:-1] + '#' + attribute
            if (node_name, attribute_name) not in Graph.edges:
                Graph.add_edge(node_name, attribute_name, type='attribute')
        for child in node.getchildren():
            child_name = node_name + '/' + child.tag
            if (node_name, child_name) not in Graph.edges:
                Graph.add_edge(node_name, child_name, type='child')
            _recurse(Graph, child, parent_path + node.tag + '/')
        
    G = nx.DiGraph()
    xml_tree = ET.fromstring(xml_string)
    _recurse(G, xml_tree)
    return G

def generate_pairwise_graph(G1, G2):
    G = nx.DiGraph()
    for a, b in G1.edges:
        type1 = G1[a][b]['type']
        for p, q in G2.edges:
            type2 = G2[p][q]['type']
            if type1 == type2:
                G.add_edge((a, p), (b, q))
                G.add_edge((b, q), (a, p))
    G = add_weights_to_pairwise(G)
    return G

def add_weights_to_pairwise(G):
    for node in G.nodes():
        G.nodes[node]['score'] = 1
        degree = len(G[node])
        edges = [(node, adjacent) for adjacent in G[node]]
        for adjacent in G[node]:
            G[node][adjacent]['weight'] = 1.0 / degree
    return G

def do_one_interation_of_flooding(G):
    G2 = G.copy()
    for (u, v, d) in G.edges(data=True):
        G2.nodes[v]['score'] += G.nodes[u]['score'] * d['weight']
    all_scores = nx.get_node_attributes(G2, name='score')
    max_score = max(all_scores.values())
    
    for node, data in G2.nodes(data=True):
        G2.nodes[node]['score'] = data['score'] / max_score
    return G2

class SimilarityFlooding:
    def __init__(self, xml_1, xml_2):
        self.graph_1 = generate_graph(xml_1)
        self.graph_2 = generate_graph(xml_2)
        self.pairwise_graph = generate_pairwise_graph( \
            self.graph_1, self.graph_2)
        self.initial_weights =  None
    
    def set_initial_scores(self, score_matrix):
        for row in score_matrix.index:
            for column in score_matrix.index:
                self.pairwise_graph.nodes[(row, column)]['score'] = \
                    score_matrix.loc[row, column]
    
    def flood_once(self):
        """
        Returns the average of the change of score in each node
        """
        initial_scores = {k: v['score'] for k, v in \
            dict(self.pairwise_graph.nodes(data=True)).items()}
        self.pairwise_graph = do_one_interation_of_flooding(self.pairwise_graph)
        final_scores = {k: v['score'] for k, v in \
            dict(self.pairwise_graph.nodes(data=True)).items()}
        return sum(abs(initial_scores[key] - final_scores[key]) \
            for key in initial_scores) / len(initial_scores.values())