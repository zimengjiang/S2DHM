bsub -n 10 -W 120:00 -wt 10 -wa INT -R "rusage[mem=10000, ngpus_excl_p=1]" python run.py \
	--dataset robotcar \
	--mode sparse_to_dense \
	--out_fname ../results/robotcar/scratch_intrinsics.txt \
	--vgg_ckpt ../checkpoints/robotcar/train_from_scratch_1e-5_3_checkpoint.pth.tar
