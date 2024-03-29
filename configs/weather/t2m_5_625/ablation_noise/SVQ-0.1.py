method = 'SimVPVQ'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 256
N_T = 8
N_S = 2
# training
lr = 5e-3
batch_size = 16
drop_path = 0.1
sched = 'cosine'

warmup_epoch = 5
epoch = 50
patience = 5
loss = 'mae'

# main setting
post_vq = False
vq_type = 'svq'
vq_kwargs = {'input_dim': hid_S,
             'freeze_codebook': False,
             'codebook_size': 10000,
             'nonlinear': True,
             'middim': 128
        }
noise = 0.1