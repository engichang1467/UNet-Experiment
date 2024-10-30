from pathlib import Path
from dataloader_spacetime import RB2DataLoader
import torch
from torch.utils.data import DataLoader
from models.baselines.U_net import U_net, UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Prepare data
data_src = Path.cwd().joinpath('output_snapshots/rb2d_ra1e6_s1.npz')

data_loader = RB2DataLoader(nt=16, 
                            data_filename=data_src, 
                            n_samp_pts_per_crop=10000, 
                            downsamp_t=2, 
                            downsamp_xz=2, 
                            return_hres=True)

data_batches = DataLoader(data_loader, batch_size=16, shuffle=True, num_workers=1)


# Initialize the model
model = UNet(in_channels=2, out_channels=1).to(device)  # Adjust channels as per your data

# Or use this one

# # time_range = 6
# output_length = 4
# input_length = 25
# # learning_rate = 0.001
# dropout_rate = 0.0
# kernel_size = 3
# batch_size = 16 # 32
# num_epoch = 1 # 1000
# # coef = 0.0
# # decay_rate = 0.95
# inp_dim = 2

# model = U_net(input_channels = inp_dim, 
#               output_channels = inp_dim, 
#               kernel_size = kernel_size, 
#               dropout_rate = dropout_rate).to(device)


# Dummy prediction
model.eval()
with torch.no_grad():
    # for batch in data_loader:
    for batch in data_batches:
        hires_input_batch, lowres_input_batch, point_coords, point_values = batch
        
        # hires_input_batch = hires_input_batch.to(device)

        # Forward pass to get prediction
        prediction = model(hires_input_batch)
        print("Dummy prediction:", prediction)
        break  # Remove this line if you want predictions on the entire dataset