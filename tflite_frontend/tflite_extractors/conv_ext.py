import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.ops.convolution import Convolution
from mo.utils.error import Error

class Conv2DFrontExtractor(FrontExtractorOp):
    op = 'CONV_2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        # Extract pads attribute
        # In case if pads is not specified it will be set in default (1) in infer function
        pads = node.tf_padding
        assert pads is None or len(pads) % 2 == 0
        # print(pads)
        auto_pad = 'SAME'
        if len(pads) == 2:
            auto_pad = 'valid'
            pads.extend([0,0])
        pads = np.array(pads, dtype=np.int64)
        # if tflite model use 'SAME' padding, pads' will look like [x, x, x, x]. if use 'VALID', pads will look like [0, 0]
        # now only handling these 2 cases!!!!!
        final_pad = None
        if pads is not None:
            pads = pads.reshape([2, -1])
            pads = np.transpose(pads)
            final_pad = np.array([[0, 0], *pads, [0, 0]], dtype=np.int64)

            # [modify data layout] #
            final_pad = np.array([[0, 0], [0, 0], *pads], dtype=np.int64)

        dilations = np.array(node.tf_dilation)  # node.dilation be like list [1, 1]
        final_dilations = np.array([1, *dilations, 1], dtype=np.int64) if dilations is not None else None  # ex. [1 1 1 1]
        # [modify data layout] #
        final_dilations = np.array([1, 1, *dilations], dtype=np.int64) if dilations is not None else None


        strides = np.array(node.tf_strides)  # node.strides be like list [1, 1]
        final_strides = np.array([1, *strides, 1], dtype=np.int64) if strides is not None else None  # ex. [1 1 1 1] 
        # [modify data layout] #
        final_strides = np.array([1, 1, *strides], dtype=np.int64) if strides is not None else None

        kernel_shape = np.array(node.tf_kernel_size, dtype=np.int64)  # ex. [5 5]  

        group = np.array(node.tf_groups, dtype=np.int64)

        # print(final_pad)
        # print(final_dilations)
        # print(final_strides)
        # print(kernel_shape)      
        # print(group)

        # print(group, kernel_shape)


        attrs = {
            'op': __class__.op,
            'type': 'Convolution',
            # 'auto_pad': auto_pad,  # mine.  # I comment this because padding is calculated in tflite_parser
            'bias_addable': True,
            'bias_term': None,
            'pad': final_pad,
            'pad_spatial_shape': np.array(pads, dtype=np.int64) if pads is not None else None,
            'dilation': final_dilations,
            'output_spatial_shape': None,
            'output_shape': None,
            'stride': final_strides,
            'group': group,
            'output': None,
            'kernel_spatial': np.array(kernel_shape, dtype=np.int64) if kernel_shape is not None else None,

            # 'input_feature_channel': 2,  # mine. set this like tf extractor's conv2d 
            # 'output_feature_channel': 3,  # mine. set this like tf extractor's conv2d 
            'input_feature_channel': 1, 
            'output_feature_channel': 0,
            'kernel_spatial_idx': None,  # Will be calculated in infer function (np.array([2, 3]))

            'spatial_dims': None,  # Will be calculated in infer function
            # 'channel_dims': np.array([3], dtype=np.int64),  # mine
            'channel_dims': np.array([1], dtype=np.int64),  # mine

            'batch_dims': np.array([0], dtype=np.int64),
            # 'layout': 'NHWC'  # mine
            'layout': 'NCHW',  # mine
        }

        # update the attributes of the node
        Convolution.update_node_stat(node, attrs)
        return cls.enabled