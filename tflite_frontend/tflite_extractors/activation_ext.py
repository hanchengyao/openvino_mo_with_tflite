from extensions.ops.activation_ops import *
from mo.front.extractor import FrontExtractorOp


class ReLUExtractor(FrontExtractorOp):
    op = 'ReLU'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU.update_node_stat(node)
        return cls.enabled

class ReLU6Extractor(FrontExtractorOp):  # map relu6 to openvino relu
    op = 'ReLU6'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReLU.update_node_stat(node)
        return cls.enabled