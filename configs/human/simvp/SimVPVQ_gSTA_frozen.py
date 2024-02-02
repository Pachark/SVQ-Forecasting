method = 'SimVPVQ'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 512
N_T = 6
N_S = 4
# training
lr = 0.00025
batch_size = 4
drop_path = 0.1
sched = 'cosine'

warmup_epoch = 0
epoch = 50
patience = 50
val_batch_size = 4
num_workers=4
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