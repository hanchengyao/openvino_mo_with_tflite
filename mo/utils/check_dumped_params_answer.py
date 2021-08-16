
import os
import numpy as np

from mo.graph.graph import Node

# [Eason] file naming cannot contain invalid char
def valid_file_name_for_const_node(const_node_name):
    invalid_chars_in_name = ['/', ':', '*', '?', '<', '>']

    for invalid_char in invalid_chars_in_name:
        if invalid_char in const_node_name:
            const_node_name = '_'.join(const_node_name.split(invalid_char))

    return const_node_name

def check_dumped_params_answer(graph, bin_file, dump_numpy_dir):
    for node in graph.nodes:
        node = Node(graph, node)
        if node.kind == 'op' and node.type == 'Const':
            npy_file_name_prefix = valid_file_name_for_const_node(node.name)
            offset = node['offset']
            size = node['size']
            npy_file = os.path.join(dump_numpy_dir, npy_file_name_prefix + '_' + str(offset) + '_' + str(size) + '.npy')
            npy_value = np.load(npy_file)
            bin_value = np.fromfile(bin_file, dtype=node.data_type, offset=offset, count=size // node.value.dtype.itemsize)
            assert np.array_equal(npy_value.flatten(), bin_value), "Const {}'s .npy file value mismatch with its correspondence in .bin file".format(node.name)
