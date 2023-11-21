# train DA object detectors (initialized by our pretrained RegionCLIP)

# RN50, cityscapes to foggy cityscapes
python3 ./tools/train_net.py \
--num-gpus 8 \
--config-file ./configs/PascalVOC-Detection/da_clip_faster_rcnn_R_50_C4.yaml \
MODEL.WEIGHTS ./output/model_lora_diff_58.3mAP.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/cityscapes_8_cls_emb.pth \
LEARNABLE_PROMPT.CTX_SIZE 8 \
LEARNABLE_PROMPT.LoRA True \
LEARNABLE_PROMPT.TUNING True #False #True 
