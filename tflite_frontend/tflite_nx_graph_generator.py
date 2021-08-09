from mo.graph.graph import Graph, Node, fill_graph_with_nodes
import logging as log
from mo.utils.error import Error, FrameworkError


def add_tflite_data_and_input_nodes(graph, data_dict, model_input_dict):
    # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
    data_nodes_map = {}
    for input_name, input_attrs in model_input_dict.items():
        graph.add_node(input_name, kind='op', op='Parameter', **input_attrs)
        data_nodes_map[input_name] = (input_name, 0)

    for data_name, data_attrs in data_dict.items():
        graph.add_node(data_name, kind='op', op='Const', **data_attrs)
        data_nodes_map[data_name] = (data_name, 0)

    return data_nodes_map

def tflite_nx_graph_generator(graph, ordered_ops_list, data_dict, model_input_dict, model_output_dict):
    # graph = Graph()
    graph.__setattr__('name', 'tflite_model')
    graph.graph['layout'] = 'NCHW'
    # graph.graph['layout'] = 'NHWC'
    graph.graph['fw'] = 'onnx'
    graph.graph['feature_dim'] = 1
    graph.graph['fw_opset_version'] = None

    # add data and input nodes (input, weight, bias, reshape_const, ...) to graph
    data_nodes_map = add_tflite_data_and_input_nodes(graph, data_dict, model_input_dict)
    # for k,v in data_nodes_map.items():
    #     print(k)


    # add output nodes
    output_ids = []
    for output_name, output_attrs in model_output_dict.items():
        if graph.has_node(output_name):
            log.error('Name {} of output node already exists in graph. Ignoring this output. If the output is required,'
                      ' please rename it.'.format(output_name), extra={'is_warning': True})
            continue
        else:
            # add fake node on output
            graph.add_node(output_name, kind='op', op='FakeOutput', **output_attrs)
            output_ids.append(output_name)

    
    # add model op nodes
    # an op looks like:
    # {'id': 'CONV_2D_0', 'tf_op_type': 'CONV_2D', 
    # 'in': ['input', 'MobilenetV1/Conv2d_0/weights', 'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_bias'], 
    # 'out': ['CONV_2D_0_inter_result'], 
    # 'attrs': {'kernel_size': [3, 3], 'strides': [2, 2], 'dilation': [1, 1], 'padding': [0, 0, 1, 1], 'data_layout': 'NHWC', 'channels': 32, 'kernel_layout': 'HWIO'}} 
    for op in ordered_ops_list:
        graph.add_node(op['id'], kind='op', tf_op_type=op['tf_op_type'], **op['attrs'])


        # if op['id'] == 'CONV_2D_0':
        # add incoming edges based on data_nodes_map
        for dst_port, inp in enumerate(op['in']):
            # should add edge inp --> id
            if inp not in data_nodes_map:
                if inp == '':
                    # input is omitted; most likely it corresponds to an optional input for an operator
                    continue
                else:
                    raise Error('Reference to {} is not satisfied. A node refer not existing data tensor. op node: {}', inp, op['id'])
            src_id, src_port = data_nodes_map[inp]

            assert (graph.has_node(src_id))
            edge_attrs = {
                'out': src_port,
                'in': dst_port,
                'name': inp,
                'fw_tensor_debug_info': [(src_id, inp)],
                'in_attrs': ['in', 'name'],
                'out_attrs': ['out', 'name'],
                'data_attrs': ['fw_tensor_debug_info']
            }
            graph.add_edge(src_id, op['id'], **edge_attrs)

        # add outgoing edges to data_nodes_map
        for src_port, out in enumerate(op['out']):
            if out in output_ids:
                edge_attrs = {
                    'out': src_port,
                    'in': 0,
                    'name': out,
                    'fw_tensor_debug_info': [(op['id'], out)],
                    'in_attrs': ['in', 'name'],
                    'out_attrs': ['out', 'name'],
                    'data_attrs': ['fw_tensor_debug_info']
                }
                graph.add_edge(op['id'], out, **edge_attrs)
            if out in data_nodes_map:
                log.debug("Detected reuse of blob {}.".format(out))
            data_nodes_map[out] = (op['id'], src_port)

    graph.graph['tensor_mapping'] = data_nodes_map  # save main graph tensor names mapping for Loop op parsing


    return graph
