import numpy as np

from extensions.ops.MatMul import GemmONNX
from mo.front.extractor import FrontExtractorOp


class GemmFrontExtractor(FrontExtractorOp):
    op = 'Gemm'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'transpose_a': False,
            'transpose_b': True,
            'axis': 0
        }
        GemmONNX.update_node_stat(node, attrs)
        return cls.enabled