import os


# grad_filters = ['np', 'sobel']
grad_filters = ['np']
loss_fns = ['sq', 'cauchy', 'gm', 'huber']

for loss_fn in loss_fns:
    for grad_filter in grad_filters:
        out_fname = "retrieval_{}_{}".format(grad_filter, loss_fn)
        use_sobel = '' if grad_filter == 'np' else '--use_sobel'
        prefix = "python test_featurePnP.py --dataset robotcar --ckpt ../checkpoints/robotcar/retrieval.pth.tar " 
        # print(prefix + "--out_fname {} --loss_fn {} {}".format(out_fname, loss_fn, use_sobel))
        os.system(prefix + "--out_fname {} --loss_fn {} {}".format(out_fname, loss_fn, use_sobel))
