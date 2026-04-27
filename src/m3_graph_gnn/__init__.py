"""
M3 Graph GNN — Microplastic Source Attribution Module
======================================================
Hydrological flow graph + GraphSAGE/GAT + Integrated Gradients attribution
for the Ogeechee, Savannah, and Altamaha river systems (coastal Georgia).
"""

from .infer import predict_concentration, attribute_source, get_node_info

__all__ = ["predict_concentration", "attribute_source", "get_node_info"]
