from mo.front.extractor import FrontExtractorOp
from mo.ops.softmax import Softmax


class SoftmaxExtractor(FrontExtractorOp):
    op = 'Softmax'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = node.axis
        Softmax.update_node_stat(node, {'axis': axis})
        return cls.enabled