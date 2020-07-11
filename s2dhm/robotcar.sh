CUDA_VISIBLE_DEVICES=1 python run.py \
	--dataset robotcar \
	--mode sparse_to_dense \
	--out_fname ../results/robotcar/contrastive_only_intrinsics.txt \
	--vgg_ckpt ../checkpoints/robotcar/ckpt0_only_contrastive_1e-7_marginpos01.pth.tar
