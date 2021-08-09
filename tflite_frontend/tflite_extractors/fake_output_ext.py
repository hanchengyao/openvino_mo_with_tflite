from mo.front.extractor import FrontExtractorOp
from extensions.ops.fake_output import FakeOutput


class FakeOutputExtractor(FrontExtractorOp):
    op = 'FakeOutput'
    enabled = False

    @classmethod
    def extract(cls, node):
        FakeOutput.update_node_stat(node)
        return cls.enabled