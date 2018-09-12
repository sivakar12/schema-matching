import networkx as nx
import xml.etree.ElementTree as ET

def generate_graph(xml_string):
    def _recurse(Graph, node):
        tag = node.tag
        if tag not in Graph.nodes:
            Graph.add_node(tag)
        for attribute in node.attrib.keys():
            if (node.tag, attribute) not in Graph.edges:
                Graph.add_edge(node.tag, attribute, type='attribute')
        for child in node.getchildren():
            if (node.tag, child.tag) not in Graph.edges:
                Graph.add_edge(node.tag, child.tag, type='child')
            _recurse(Graph, child)
        
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
    nx.nodes
    return G2