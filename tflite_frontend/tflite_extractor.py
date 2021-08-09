from mo.graph.graph import Node
from tflite_frontend.tflite_extractors import conv_ext, reshape_ext, activation_ext, const_ext, placeholder_ext, fake_output_ext, fully_connected_ext, gemm_ext, softmax_ext, pooling_ext

tflite_op_extractors = {
    'AVERAGE_POOL_2D': pooling_ext.AvgPoolFrontExtractor.extract,
    'CONV_2D': conv_ext.Conv2DFrontExtractor.extract,
    'Const': const_ext.ConstExtractor.extract,
    'DEPTHWISE_CONV_2D': conv_ext.Conv2DFrontExtractor.extract,
    'FakeOutput': fake_output_ext.FakeOutputExtractor.extract,
    'FULLY_CONNECTED': gemm_ext.GemmFrontExtractor.extract,  # to be verified
    'Parameter': placeholder_ext.PlaceholderFrontExtractor.extract,
    'RELU': activation_ext.ReLUExtractor.extract,
    'RELU6': activation_ext.ReLU6Extractor.extract,
    'RESHAPE': reshape_ext.ReshapeFrontExtractor.extract,
    'SOFTMAX': softmax_ext.SoftmaxExtractor.extract
}

def common_tflite_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.id,
        'op': node.op if node.has_valid('op') else node.tf_op_type 
    }

def tflite_op_extractor(node: Node):
    result = common_tflite_fields(node)
    node.graph.node[node.id].update(result)
    supported = False
    op = result['op']
    assert op in tflite_op_extractors
    tflite_op_extractors[op](node)
    result.update(node.graph.node[node.id])
    supported = True
    return supported, result