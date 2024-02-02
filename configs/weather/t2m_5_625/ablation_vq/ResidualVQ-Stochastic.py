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
patience = 10
loss = 'mse'

# main setting
post_vq = False
vq_type = 'rvq'
vq_kwargs = {'dim': hid_S,   
             'num_quantizers': 8,
    'codebook_size': 1024,   
    'stochastic_sample_codes': True,
    'sample_codebook_temp': 0.1,
    'shared_codebook': True
}