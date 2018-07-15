python ../eval_image_classifier.py \
--checkpoint_path=../log/train/vgg16 \
--eval_dir=../log/train/eval \
--dataset_name=car \
--dataset_split_name=test \
--dataset_dir=../datarecord \
--model_name=vgg_16