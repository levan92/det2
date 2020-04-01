import torch
import numpy as np
from tridentnet import add_tridentnet_config
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
# from detectron2.engine.defaults import DefaultPredictor
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
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
    # register_coco_instances("Ships", {},"ships-lite.json","")
    # register_coco_instances("Ships", {},"ships-lite.json","Ships")
    # register_coco_instances("Ships", {},"config/ships-lite.json","Ships")
    # metadata = MetadataCatalog.get("Ships")
    
    classes_path = args['classes_path']
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return cfg, class_names

class Det2(object):
    _defaults = {
        "weights": 'weights/tridentnet_finetune_2/model_0049999.pth',
        "config": 'configs/demo_tridentnet_fast_R_50_C4_1x.yaml',
        "classes_path": 'configs/PP_classes.txt',
        "thresh": 0.5,
    }

    def __init__(self, bgr=True, **kwargs):
        self.__dict__.update(self._defaults)
        # for portability between keras-yolo3/yolo.py and this
        if 'model_path' in kwargs:
            kwargs['weights'] = kwargs['model_path']
        if 'score' in kwargs:
            kwargs['thresh'] = kwargs['score']
        self.__dict__.update(kwargs)
        cfg, self.class_names = setup(self.__dict__)
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        # If not specified, input size into network is min 800 and max 1333
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        print('Model expects images in {} format, but dn worry, this object takes care of that for you, just be init it correctly when you instantiate this object'.format(self.input_format))

        # warm up
        self._detect([np.zeros((10,10,3), dtype=np.uint8)])
        print('Warmed up!')

    @torch.no_grad()
    def _detect(self, list_of_imgs):
        """
        Args:
            list of images (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model
        """
        inputs = []        
        for img in list_of_imgs:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                img = img[:, :, ::-1]
            height, width = img.shape[:2]
            image = self.transform_gen.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # print(image.size())
            inputs.append({"image": image, "height": height, "width": width})
        predictions = self.model(inputs)
        return predictions

    def _postprocess(self, preds, box_format='ltrb', wanted_classes=None, buffer_ratio=0.0):
        all_dets = []
        for pred in preds:
            pred = pred['instances'].to('cpu')
            im_height, im_width = pred.image_size
            bboxes = pred.pred_boxes.tensor.numpy()
            scores = pred.scores.numpy()
            pred_classes = pred.pred_classes.numpy()

            dets = []
            for bbox, score, class_ in zip(bboxes, scores, pred_classes):
                pred_class_name = self.class_names[class_]
                if wanted_classes is not None and pred_class_name not in wanted_classes:
                    continue
                l, t, r, b = bbox
                
                w = r - l + 1
                h = b - t + 1
                width_buffer = w * buffer_ratio
                height_buffer = h * buffer_ratio
                
                l = max( 0.0, l-0.5*width_buffer )
                t = max( 0.0, t-0.5*height_buffer )
                r = min( im_width - 1.0, r + 0.5*width_buffer )
                b = min( im_height - 1.0, b + 0.5*height_buffer )

                box_infos = []
                for c in box_format:
                    if c == 't':
                        box_infos.append( int(round(t)) ) 
                    elif c == 'l':
                        box_infos.append( int(round(l)) )
                    elif c == 'b':
                        box_infos.append( int(round(b)) )
                    elif c == 'r':
                        box_infos.append( int(round(r)) )
                    elif c == 'w':
                        box_infos.append( int(round(w+width_buffer)) )
                    elif c == 'h':
                        box_infos.append( int(round(h+height_buffer)) )
                    else:
                        assert False,'box_format given in detect unrecognised!'
                assert len(box_infos) > 0 ,'box infos is blank'

                dets.append( (box_infos, score, pred_class_name) )
            
            all_dets.append(dets)
        return all_dets


    def detect_get_box_in(self, images, box_format='ltrb', classes=None, buffer_ratio=0.):
        '''
        Params
        ------
        - images : ndarray-like or list of ndarray-like
        - box_format : string of characters representing format order, where l = left, t = top, r = right, b = bottom, w = width and h = height
        - classes : list of string, classes to focus on
        - buffer : float, proportion of buffer around the width and height of the bounding box

        Returns
        -------
        if one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class),
        
        else if a list of ndarray given, this return a list (batch) containing the former as the elements,

        where,
            - box_infos : list of floats in the given box format
            - score : float, confidence level of prediction
            - predicted_class : string

        '''
        single = False
        if isinstance(images, list):
            if len(images) <= 0 : 
                return None
            else:
                assert all(isinstance(im, np.ndarray) for im in images)
        elif isinstance(images, np.ndarray):
            images = [ images ]
            single = True

        res = self._detect(images)
        all_dets = self._postprocess(res, box_format=box_format, wanted_classes=classes, buffer_ratio=buffer_ratio)

        if single:
            return all_dets[0]
        else:
            return all_dets

if __name__ == '__main__':
    import cv2
    det2 = Det2()
    imgpath = '/media/dh/HDD1/4K_sea_scenes/DJI_0044_4K_SEA_decoded/DJI_0044_4K_SEA_frame0110.jpg'
    # imgpath = '/media/dh/HDD1/pp/someShips/4.jpg'
    img = cv2.imread(imgpath)
    # img2 = cv2.resize(img, (200,200))
    n = 100
    import time
    dur = 0
    for _ in range(n):
        tic = time.time()
        res = det2.detect_get_box_in(img, box_format='ltrb', classes=None, buffer_ratio=0.0)
        toc = time.time()
        dur += toc - tic
    print('Time taken: {:0.3f}s'.format(dur/n))

    cv2.namedWindow('', cv2.WINDOW_NORMAL)
    draw_frame = img.copy()
    for r in res:
        bb, score, class_ = r 
        l,t,r,b = bb
        cv2.rectangle(draw_frame, (l,t), (r,b), (255,255,0), 2, )
    
    cv2.imshow('', draw_frame)
    cv2.waitKey(0)