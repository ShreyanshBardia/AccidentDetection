from .YOLOv3 import YOLOv3
# from .MMDet import MMDet
from .YOLOv3.yolo_utils import get_all_boxes, nms, post_process, xywh_to_xyxy, xyxy_to_xywh
import cv2

__all__ = ['build_detector']

def build_detector(cfg, use_cuda):
    if cfg.USE_MMDET:
        return MMDet(cfg.MMDET.CFG, cfg.MMDET.CHECKPOINT,
                    score_thresh=cfg.MMDET.SCORE_THRESH,
                    is_xywh=True, use_cuda=use_cuda)
    else:
        import torch,torchvision
        maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        maskrcnn = maskrcnn.to(device)
        maskrcnn.eval()
        class mask_rcnn():
            def __call__(self,img):

                if (img>1).any():
                    img=(img/255).astype('float32')
                print(f"{img.shape=}",type(img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).permute(2, 0, 1)
                o = maskrcnn([img.to(device)])[0]
                return (xyxy_to_xywh(o['boxes']).cpu().detach().numpy(),
                o['scores'].cpu().detach().numpy(),o['labels'].cpu().detach().numpy(),o['masks'].cpu().detach().numpy())
        return mask_rcnn()
        

