from mo.front.common.replacement import FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.replacement import FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileOp, \
    FrontReplacementFromConfigFileGeneral


def get_front_classes():
    front_classes = [FrontExtractorOp, FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph,
                     FrontReplacementFromConfigFileSubGraph, FrontReplacementFromConfigFileOp,
                     FrontReplacementFromConfigFileGeneral]
    return front_classes