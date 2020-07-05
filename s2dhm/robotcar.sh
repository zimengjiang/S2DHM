CUDA_VISIBLE_DEVICES=0 python run.py \
	--dataset robotcar \
	--mode sparse_to_dense \
	--out_fname ../results/robotcar/retrieval.txt \
	--vgg_ckpt ../checkpoints/robotcar/retrieval.pth.tar
