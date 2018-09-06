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
    return G