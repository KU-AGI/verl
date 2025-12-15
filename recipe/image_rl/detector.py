import io
import base64
import argparse
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import box_convert, nms


# =============================================================================
# Pydantic Models
# =============================================================================
class InfoItem(BaseModel):
    type: str  # "spatial" or "numeracy"
    subject: Optional[str] = None
    object: Optional[str] = None
    relation: Optional[str] = None
    num: Optional[str] = None


class DetectionRequest(BaseModel):
    info_list: List[InfoItem]
    img_url: str  # base64 encoded image


class DetectionResult(BaseModel):
    det_judge: bool
    det_reason: str
    det_info: dict
    vis_data: Optional[dict] = None


class DetectionResponse(BaseModel):
    results: List[List[DetectionResult]]


# =============================================================================
# GDino Model Class
# =============================================================================
class GDino:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[GDino] Loading model from: {args.gdino_ckpt_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(args.gdino_ckpt_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.gdino_ckpt_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        print(f"[GDino] Model loaded on device: {self.device}", flush=True)

        self.box_threshold = args.box_thrs
        self.text_threshold = args.box_thrs
        self.iou_threshold = args.iou_thrs
        self.do_visualize = args.visualize
        
        self.distance_threshold_max = args.dist_thrs_max
        self.distance_threshold_min = args.dist_thrs_min 
        self.area_threshold = args.area_thrs

    @torch.no_grad()
    def __call__(self, info, labels, img):
        inputs = self.processor(
            images=img, text=labels, return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(**inputs)

        target_sizes = [(img.height, img.width)]
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes
        )[0]

        boxes_xyxy = res["boxes"]
        scores = res["scores"]
        pred_labels = res["labels"]
        
        if boxes_xyxy.numel() > 0:
            boxes_cxcywh = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
        else:
            boxes_cxcywh = torch.empty((0, 4), device=self.device)

        det_judge, reason, vis_data = False, "No detections", None
        info_type = info.get("type", "")

        try:
            if boxes_cxcywh.numel() > 0:
                if info_type == "spatial":
                    det_judge, reason, vis_data = self.get_spatial_score(
                        boxes_cxcywh, pred_labels, scores, info
                    )
                elif info_type == "numeracy":
                    det_judge, reason, vis_data = self.get_numeracy_score(
                        boxes_cxcywh, pred_labels, scores, info
                    )
        except Exception as e:
            reason = f"Error: {e}"

        return [{
            "det_judge": det_judge, 
            "det_reason": str(reason),
            "det_info": info,
            "vis_data": vis_data
        }]

    def get_spatial_score(self, boxes_cxcywh, labels, scores, info):
        obj1, obj2 = info['subject'], info['object']
        
        idx1 = [i for i, label in enumerate(labels) if label == obj1 or obj1 in label]
        idx2 = [i for i, label in enumerate(labels) if label == obj2 or obj2 in label]
        
        vis_payload, flag = None, False
        reason = "Missing Object"
        b1_best, b2_best = None, None
        
        if idx1 and idx2:
            best_idx1 = idx1[torch.argmax(scores[torch.tensor(idx1)]).item()]
            best_idx2 = idx2[torch.argmax(scores[torch.tensor(idx2)]).item()]
            
            b1_best = boxes_cxcywh[best_idx1].tolist()
            b2_best = boxes_cxcywh[best_idx2].tolist()
            
            flag, reason = self.determine_position(info['relation'], b1_best, b2_best)
        else:
            if not idx1:
                reason = f"Missing Subject: {obj1}"
            elif not idx2:
                reason = f"Missing Object: {obj2}"

        if self.do_visualize:
            vis_payload = {
                'type': 'spatial',
                'box1': b1_best, 'box2': b2_best, 'info': info, 'result': flag
            }
        return flag, reason, vis_payload

    def get_numeracy_score(self, boxes_cxcywh, labels, scores, info):
        target_expr, obj_name = info['num'], info['object']
        
        target_indices = [i for i, label in enumerate(labels) if label == obj_name or obj_name in label]
        
        detected_count = 0
        tgt_boxes_vis = []
        
        if target_indices:
            tgt_indices_tensor = torch.tensor(target_indices, device=self.device)
            tgt_boxes = boxes_cxcywh[tgt_indices_tensor]
            tgt_scores = scores[tgt_indices_tensor]
            
            tgt_xyxy = box_convert(tgt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            keep = nms(tgt_xyxy, tgt_scores, self.iou_threshold)
            
            detected_count = len(keep)
            tgt_boxes_vis = tgt_boxes[keep].tolist()

        flag = False
        try:
            val = int(target_expr)
            flag = (detected_count == val)
        except:
            s = str(target_expr).strip()
            try:
                if s.startswith(">="):
                    flag = (detected_count >= int(s[2:]))
                elif s.startswith("<="):
                    flag = (detected_count <= int(s[2:]))
                elif s.startswith("=="):
                    flag = (detected_count == int(s[2:]))
                elif s.startswith(">"):
                    flag = (detected_count > int(s[1:]))
                elif s.startswith("<"):
                    flag = (detected_count < int(s[1:]))
                else:
                    flag = False
            except:
                flag = False

        vis_payload = None
        if self.do_visualize:
            vis_payload = {
                'type': 'counting',
                'boxes': tgt_boxes_vis, 'info': info, 'result': flag
            }
            
        return flag, f"Count: {detected_count} (Exp: {target_expr})", vis_payload

    def determine_position(self, locality, box1, box2):
        b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
        b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
        b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
        b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = box1[2] * box1[3]
        
        subj_in_obj = (b2_x1 <= b1_x1 and b2_x2 >= b1_x2 and b2_y1 <= b1_y1 and b2_y2 >= b1_y2)
        
        cx1, cy1 = box1[0], box1[1]
        cx2, cy2 = box2[0], box2[1]
        
        off_x = cx1 - cx2
        off_y = cy1 - cy2
        
        score, reason = False, "Init"
        
        thrs_val = self.distance_threshold_max
        penalty_x = thrs_val * (box1[2] + box2[2])
        penalty_y = thrs_val * (box1[3] + box2[3])
        
        rev_x = max(abs(off_x) - penalty_x, 0) * (1 if off_x >= 0 else -1)
        rev_y = max(abs(off_y) - penalty_y, 0) * (1 if off_y >= 0 else -1)

        norm = (rev_x**2 + rev_y**2)**0.5 + 1e-6
        dx, dy = rev_x / norm, rev_y / norm

        if locality in ['left', 'right', 'top', 'bottom', 'above', 'below']:
            if abs(rev_x) < 1e-5 and abs(rev_y) < 1e-5:
                return False, "Too close"
            
            if locality == 'left':
                score = (dx < -0.5)
            elif locality == 'right':
                score = (dx > 0.5)
            elif locality in ['top', 'above']:
                score = (dy < -0.5)
            elif locality in ['bottom', 'below']:
                score = (dy > 0.5)
            reason = f"Directional (dx={dx:.2f}, dy={dy:.2f})"
            
        elif locality in ['near', 'next to', 'close']:
            limit_x = (box1[2]+box2[2])/2 + self.distance_threshold_min
            limit_y = (box1[3]+box2[3])/2 + self.distance_threshold_min
            score = (abs(off_x) < limit_x or abs(off_y) < limit_y)
            reason = "Proximity"
            
        elif locality in ['inside', 'in']:
            score = (inter_area >= (area1 * self.area_threshold)) or subj_in_obj
            reason = "Overlap/Containment"

        return score, reason


# =============================================================================
# Global
# =============================================================================
gdino_model: Optional[GDino] = None
args = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdino_ckpt_path", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box_thrs", type=float, default=0.25)
    parser.add_argument("--iou_thrs", type=float, default=0.5)
    parser.add_argument("--dist_thrs_max", type=float, default=0.3)
    parser.add_argument("--dist_thrs_min", type=float, default=0.1)
    parser.add_argument("--area_thrs", type=float, default=0.5)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gdino_model, args
    print(f"[Server] Initializing GDino model...", flush=True)
    print(f"[Server] Model path: {args.gdino_ckpt_path}", flush=True)
    gdino_model = GDino(args)
    print(f"[Server] GDino model loaded successfully!", flush=True)
    yield
    print(f"[Server] Shutting down...", flush=True)


app = FastAPI(
    title="GDino Object Detection API",
    description="Zero-shot object detection with Grounding DINO",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# Utility Functions
# =============================================================================
def convert_base64_to_img_pil(base64_str: str) -> Image.Image:
    if not isinstance(base64_str, str):
        raise TypeError("base64_str must be a string")
    
    base64_str = base64_str.strip()
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[-1]
    
    image_bytes = base64.b64decode(base64_str, validate=False)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.load()
    return image


def add_article(word: str) -> str:
    if word and word[0].lower() in 'aeiou':
        return f"an {word}"
    return f"a {word}"


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": gdino_model is not None}


@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    if gdino_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        img_pil = convert_base64_to_img_pil(request.img_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    results = []
    for info in request.info_list:
        info_dict = info.model_dump()
        info_type = info_dict.get("type", "")
        
        labels = []
        if info_type == "spatial":
            labels = [
                add_article(info_dict['subject'].lower()),
                add_article(info_dict['object'].lower())
            ]
        elif info_type == "numeracy":
            labels = [add_article(info_dict['object'].lower())]
        
        result = gdino_model(info_dict, labels, img_pil)
        results.append(result)
    
    return {"results": results}


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    args = get_args()
    uvicorn.run(app, host=args.host, port=args.port)