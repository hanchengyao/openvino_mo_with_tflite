from extensions.ops.MatMul import FullyConnected
from mo.front.extractor import FrontExtractorOp


class FullyConnectedFrontExtractor(FrontExtractorOp):
    op = 'FullyConnected'
    enabled = True

    @classmethod
    def extract(cls, node):
        FullyConnected.update_node_stat(node)
        return cls.enabled