1. Put your dataset at './datasets/your_dataset'. Please follow the format of Pascal Voc.
   For example:

- dataset
  - cityscapes_voc
    - VOC2007
      - Annotations
      - ImageSets
      - JPEGImages
  - foggy_cityscapes_voc
    - VOC2007
      - Annotations
      - ImageSets
      - JPEGImages

2. Put your pre-trained VLM model at somewhere you like, for example, './ckpt', and edit the MODEL.WEIGHTS in train_da_ada_c2f.sh.
3. Following RegionCLIP, generate class embedding and put it at somewhere you like, and edit the MODEL.CLIP.TEXT_EMB_PATH.
4. Training: train_da_ada_c2f.sh  Testing: test_da_ada_c2f.sh
