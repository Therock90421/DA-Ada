# visualize detection results from finetuned detectors on custom images

########################################################

# Open-vocabulary detector trained by 866 LVIS base categories, with RegionCLIP (RN50x4) as initialization
python3 ./tools/train_net.py \
--eval-only \
--num-gpus 1 \
--config-file ./configs/PascalVOC-Detection/da_clip_faster_rcnn_R_50_C4.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-coco_rn50.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/cityscapes_8_cls_emb.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
LEARNABLE_PROMPT.CTX_SIZE 8 \
LEARNABLE_PROMPT.TUNING False 

# visualize the prediction json file
#python ./tools/visualize_json_results.py \
#--input ./output/inference/lvis_instances_results.json \
#--output ./output/regions \
#--dataset lvis_v1_val_custom_img \
#--conf-threshold 0.05 \
#--show-unique-boxes \
#--max-boxes 25 \
#--small-region-px 8100\


########################################################



