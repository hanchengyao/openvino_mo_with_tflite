import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph

# from extensions.back.GroupedConvWeightsNormalize import GroupedConvWeightsNormalize
from extensions.back.pass_separator import BackFinish
 

class ConvActFusion(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [BackFinish]

    def pattern(self):
        return dict(
            nodes = [
                ('conv', dict(kind = 'op', type = lambda tp: tp in ['Convolution', 'GroupConvolution'])),
                ('conv_result', dict(kind = 'data')),
                ('act_func', dict(kind = 'op', type = lambda tp: tp in ['ReLU', 'ReLU6', 'SoftMax'])),
                ('act_func_result', dict(kind= 'data')),
            ],

            edges = [
                ('conv', 'conv_result'),
                ('conv_result', 'act_func'),
                ('act_func', 'act_func_result')
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        # print("in my pass")
        match['conv'].out_port(0).disconnect()
        match['act_func'].out_port(0).get_connection().set_source(match['conv'].out_port(0))

        # add 'act_func' attr to conv node
        match['conv'].__setitem__('act_func', match['act_func'].op)