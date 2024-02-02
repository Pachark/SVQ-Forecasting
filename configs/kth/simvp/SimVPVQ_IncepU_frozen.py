method = 'SimVPVQ'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # SimVP.V1
hid_S = 64
hid_T = 256
N_T = 6
N_S = 2
# training
lr = 5e-4 # 1e-3
drop_path = 0.1
batch_size = 4  # bs = 2 x 8GPUs
sched = 'onecycle'
val_batch_size = 4

warmup_epoch = 0
epoch = 100
patience = 50
loss = 'mae'

# main setting
post_vq = False
vq_type = 'svq'
vq_kwargs = {'input_dim': hid_S,
             'freeze_codebook': True,   
             'codebook_size': 6000,   
             'init_method': 'kaiming',
             'nonlinear': True,
             'middim': 128
        }