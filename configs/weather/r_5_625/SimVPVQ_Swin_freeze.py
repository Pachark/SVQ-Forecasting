method = 'SimVPVQ'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'swin'
hid_S = 32
hid_T = 256
N_T = 8
N_S = 2
# training
lr = 1e-3
batch_size = 16
drop_path = 0.2
sched = 'cosine'

warmup_epoch = 5

epoch = 50
patience = 10
loss = 'mae'

# main setting
post_vq = False
vq_type = 'svq'
vq_kwargs = {'input_dim': hid_S,
             'freeze_codebook': True,   
             'codebook_size': 10000,   
             'init_method': 'kaiming',
             'nonlinear': True,
             'middim': 128
        }