import numpy as np

from extensions.ops.parameter import Parameter
from mo.front.extractor import FrontExtractorOp


class PlaceholderFrontExtractor(FrontExtractorOp):
    op = 'Parameter'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'shape': np.array(node.shape, dtype=np.int64),
            'data_type': node.dtype
        }
        Parameter.update_node_stat(node, attrs)
        return cls.enabled