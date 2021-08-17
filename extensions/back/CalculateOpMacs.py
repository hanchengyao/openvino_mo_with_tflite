from mo.graph.graph import Graph, Node
from mo.back.replacement import BackReplacementPattern
from extensions.back.pass_separator import BackFinish

from extensions.back.CalculateTotalMacs import CalculateTotalMacs

import numpy as np

macs_accountable_ops = {'Convolution', 'GroupConvolution', 'MatMul', 'ConvolutionBackpropData', 'GroupConvolutionBackpropData'}

data_movement_ops = {'Broadcast', 'Concat', 'Gather', 'GatherElements', 'GatherND', \
    'Reverse', 'ReverseSequence', 'Roll', 'ScatterElementsUpdate', 'ScatterNDUpdate', \
        'Select', 'ScatterUpdate', 'Split', 'StridedSlice', 'ShuffleChannels', \
            'SpaceToBatch', 'SpaceToDepth', 'Transpose', 'Tile', 'VariadicSplit'}

def calculate_op_macs(node: Node):
    # Compute macs for [2-d] Conv and ConvTranspose
    if node.type == 'Convolution' or node.type == 'ConvolutionBackpropData':
        # according to openvino's doc, node.in/out_node... is a legacy approach and is not recommended. 
        input_shape = node.in_port(0).data.get_shape()  
        weight_shape = node.in_port(1).data.get_shape()
        output_shape = node.out_port(0).data.get_shape()
        if len(weight_shape) == 4:  # now only implement macs computation for 2-d convolution.         
            # Convolution: macs = batch_size x K × K × Cin × Hout × Wout × Cout
            node['macs'] = input_shape[0] * weight_shape[2] * weight_shape[3] * input_shape[1] * output_shape[2] * output_shape[3] * output_shape[1]

    # Compute macs for [2-d] Depthwise Conv and dw ConvTranspose
    elif node.type == 'GroupConvolution' or node.type == 'GroupConvolutionBackpropData':
        input_shape = node.in_port(0).data.get_shape()
        weight_shape = node.in_port(1).data.get_shape() # 2-d GroupConv weight layout: [GROUPS, C_OUT for each group, C_IN, Y, X]
        output_shape = node.out_port(0).data.get_shape()
        if weight_shape[0] == input_shape[1] and len(weight_shape) == 5:  # means 2-d dw
            # Depthwise Convolution: macs = batch_size x output_multiplier x K × K × Cin × Hout × Wout
            node['macs'] = input_shape[0] * weight_shape[1] * weight_shape[3] * weight_shape[4] * input_shape[1] * output_shape[2] * output_shape[3]

    # Compute macs for MatMul (including Fully-connected op since it is transfomed to MatMul)
    elif node.type == 'MatMul':
        input_shape = node.in_port(0).data.get_shape()
        weight_shape = node.in_port(1).data.get_shape()
        node['macs'] = np.prod(input_shape[:-1]) * np.prod(weight_shape)



# [Eason]
# making this pass run after BackFinish to calculate macs for the final graph after all transformation
# transformation type: generic middle pass. If enabled is true, this pass will be run.
# this pass also calculate macs_ops/total_ops and data_movement_ops/total_ops
class CalculateOpMacs(BackReplacementPattern):
    enabled = True

    def run_after(self):
        return [BackFinish]

    def run_before(self):
        return [CalculateTotalMacs]

    def find_and_replace_pattern(self, graph: Graph):
        total_ops_cnt = 0
        macs_ops_cnt = 0
        data_movement_ops_cnt = 0
        total_params = 0
        total_activations_size = 0
        total_macs = 0

        not_operator_op_types = {'Const', 'Parameter', 'Result'}

        for node in graph.nodes:
            node = Node(graph, node)
            if node.kind == 'op' and node.type not in not_operator_op_types:
                total_ops_cnt += 1

                # sum up all activations' size
                if node.has_valid('act_func'):
                    node_act_shape = node.out_port(0).data.get_shape()
                    total_activations_size += np.prod(node_act_shape)

                # calculate op macs
                if node.type in macs_accountable_ops:
                    macs_ops_cnt += 1
                    calculate_op_macs(node)
                    total_macs += node['macs']

                elif node.type in data_movement_ops:
                    data_movement_ops_cnt += 1

            # calculate number of params
            elif node.kind == 'op' and node.type == 'Const':
                total_params += node.value.size

        graph.graph['total_macs'] = str(round(total_macs / 10**6, 2)) + 'M'

        graph.graph['macs_ops_proportion'] = str(round(macs_ops_cnt / total_ops_cnt * 100, 2)) + '%'
        graph.graph['data_movement_ops_proportion'] = str(round(data_movement_ops_cnt / total_ops_cnt * 100, 2)) + "%"

        graph.graph['params_size'] = str(round(total_params / 10**6, 2)) + 'M'

        graph.graph['activations_size'] = str(round(total_activations_size / 10**6, 2)) + 'M'
