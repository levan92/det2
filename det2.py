from tridentnet import add_tridentnet_config

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets import register_coco_instances

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg_file = args['config']
    model_weights = args['weights']
    positive_thresh = args['thresh']

    cfg = get_cfg()
    add_tridentnet_config(cfg)
    cfg.merge_from_file(cfg_file)

    # This is the way we set the thresh, and model weights. TODO: take in as params from args
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = positive_thresh
    cfg.MODEL.WEIGHTS = model_weights

    cfg.freeze()
    register_coco_instances("Ships", {},"ships-lite.json","Ships")
    metadata = MetadataCatalog.get("Ships")

    return cfg, metadata

class Det2(object):
    _defaults = {
        "weights": 'weights/tridentnet_finetune_2/model_0049999.pth',
        "config": 'configs/demo_tridentnet_fast_R_50_C4_1x.yaml',
        "thresh": 0.5
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        cfg, meta = setup(self.__dict__)
        predictor = DefaultPredictor(cfg)



if __name__ == '__main__':
    det2 = Det2()
