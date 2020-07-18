# for CMU_SLICE in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
# for CMU_SLICE in 3 6 11 12 16
# for CMU_SLICE in 11
# for CMU_SLICE in 10
# for CMU_SLICE in 3 6 8 10 11 12
for CMU_SLICE in 13 14
do
bsub -n 10 -W 120:00 -wt 10 -wa INT -R "rusage[mem=10000, ngpus_excl_p=1]" python run.py \
	--dataset cmu \
	--mode sparse_to_dense \
	--cmu_slice $CMU_SLICE \
	--vgg_ckpt ../checkpoints/cmu/retrieval.pth.tar
done
