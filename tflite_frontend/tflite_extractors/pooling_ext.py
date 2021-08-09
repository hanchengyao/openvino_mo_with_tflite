import logging as log

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_autopad
from mo.ops.pooling import Pooling
from mo.utils.error import Error


def common_tflite_pool_extractor(node, pool_method):
    kernel_shape = np.array(node.pool_size, dtype=np.int64)
    final_kernel_shape = np.array([1, *[x for x in kernel_shape], 1], dtype=np.int64) if kernel_shape is not None else None
    # [modify data layout] #
    final_kernel_shape = np.array([1, 1, *[x for x in kernel_shape]], dtype=np.int64) if kernel_shape is not None else None


    pads = node.padding
    assert pads is None or len(pads) % 2 == 0
    # print(pads)
    if len(pads) == 2:  # len(pads)==2 means 'valid' padding in tflite.
        pads.extend([0,0])
    pads = np.array(pads, dtype=np.int64)
    final_pads = None
    if pads is not None:
        pads = pads.reshape([2, -1])
        pads = np.transpose(pads)
        final_pads = np.array([[0, 0], *[p for p in pads], [0, 0]], dtype=np.int64)

        # [modify data layout] #
        final_pads = np.array([[0, 0], [0, 0], *[p for p in pads]], dtype=np.int64)

    strides = np.array(node.strides)  # node.strides be like list [1, 1]
    final_strides = np.array([1, *strides, 1], dtype=np.int64) if strides is not None else None  # ex. [1 1 1 1]
    # [modify data layout] #
    final_strides = np.array([1, 1, *strides], dtype=np.int64) if strides is not None else None  # ex. [1 1 1 1]


    # dilation not implemented yet. not sure if tflite's pooling has attr dilation #
    # dilation = None

    # print(final_pads)
    # print(final_strides)
    # print(kernel_shape)


    attrs = {
        'op': node.op,
        # 'auto_pad': auto_pad,
        'window': final_kernel_shape,
        'stride': final_strides,
        'pad': final_pads,
        'pad_spatial_shape': np.array(pads, dtype=np.int64) if pads is not None else None,
        'pool_method': pool_method,
        # 'exclude_pad': True if exclude_pad else False,
        # 'global_pool': global_pooling,
        'output_spatial_shape': None,
        # 'rounding_type': rt,

        'spatial_dims': None,
        # 'channel_dims': np.array([3], dtype=np.int64),
        'channel_dims': np.array([1], dtype=np.int64),

        'batch_dims': np.array([0], dtype=np.int64),
        # 'layout': 'NHWC',
        'layout': 'NCHW'

        # 'pooling_convention': pooling_convention
    }
    return attrs






class AvgPoolFrontExtractor(FrontExtractorOp):
    op = 'AveragePool'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = common_tflite_pool_extractor(node, 'avg')
        Pooling.update_node_stat(node, attrs)
        return cls.enabled