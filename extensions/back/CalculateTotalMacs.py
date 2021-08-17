from mo.graph.graph import Graph
from mo.back.replacement import BackReplacementPattern



class CalculateTotalMacs(BackReplacementPattern):
    enabled = False

    def run_after(self):
        return []

    def run_before(self):
        return []

    # this pass sums up macs of all ops and sets it as a graph's attribute, and doesn't transform the graph
    def find_and_replace_pattern(self, graph: Graph):
        total_macs = 0
        for _, attrs in graph.nodes(data=True):
            if 'macs' in attrs:
                total_macs += attrs['macs']
        graph.graph['total_macs'] = str(total_macs)