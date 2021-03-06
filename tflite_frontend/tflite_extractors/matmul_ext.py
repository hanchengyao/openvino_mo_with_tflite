from extensions.ops.MatMul import MatMul
from mo.front.extractor import FrontExtractorOp


class MatMulFrontExtractor(FrontExtractorOp):
    op = 'MatMul'
    enabled = True

    @classmethod
    def extract(cls, node):
        MatMul.update_node_stat(node)
        return cls.enabled