from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


# [Eason]
# making this a middle pass to make it run after Partial Inference
# transformation type: generic middle pass. If enabled is true, this pass will be run.
class CalTotalMacs(MiddleReplacementPattern):
    enabled = True

    # this pass sums up macs of all ops and sets it as a graph's attribute, and doesn't transform the graph
    def find_and_replace_pattern(self, graph: Graph):
        total_macs = 0
        for _, attrs in graph.nodes(data=True):
            if 'macs' in attrs:
                total_macs += attrs['macs']
        graph.graph['total_macs'] = str(total_macs)