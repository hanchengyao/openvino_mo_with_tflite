import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.ops.reshape import Reshape

class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'Reshape'
    enabled = True

    @classmethod
    def extract(cls, node):
        Reshape.update_node_stat(node)
        return cls.enabled