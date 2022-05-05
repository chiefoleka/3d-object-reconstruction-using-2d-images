import torch
import numpy as np
import time
import torch.optim as optim
from tqdm import tqdm, trange
import utils
import models

cube_len = 32
epoch_count = 10
batch_size = 16
noise_size = 256 # the length of the encoder output
d_lr = 0.00005 # discriminator learning rate
g_lr = 0.0025 # generator learning rate
log_folder = "logs/"

condition_count = 2

from torch.utils.data import TensorDataset, DataLoader

all_models1 = utils.load_all("data/chair_all", contains = '_1.mat') # names ends with a rotation number for 12 rotations, 30 degrees each
all_models7 = utils.load_all("data/chair_all", contains = '_7.mat') # 1 and 7 are 0 and 180 degrees respectively

train_set1 = torch.from_numpy(all_models1).float()
train_set7 = torch.from_numpy(all_models7).float()

train_set_all = TensorDataset(train_set1, train_set7)
train_loader = DataLoader(dataset=train_set_all, batch_size=batch_size, shuffle=True, pin_memory=True)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device("cuda:0")

generator = models.Generator(noise_size=(noise_size + 1), cube_resolution=cube_len) # noise size +1 condition value
discriminator = models.Discriminator(cube_resolution=cube_len)

generator = generator.to(device)
discriminator = discriminator.to(device)

optimizerD = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))

from torch.autograd import Variable
criterion_GAN = torch.nn.BCELoss()

def get_gan_loss(tensor,ones):
    if(ones):
        return criterion_GAN(tensor,Variable(torch.ones_like(tensor.data).to(device), requires_grad=False))
    else:
        return criterion_GAN(tensor,Variable(torch.zeros_like(tensor.data).to(device), requires_grad=False))

def get_noise(b_size = batch_size):
    return torch.randn([b_size,noise_size], device=device)

def train_GAN_epoch():
    
    g_loss = []
    d_loss = []
    gen_out = []
    train_disc = True
    
    for i, data_c in enumerate(train_loader):
        
        acc_list = []
        
        for c in range(condition_count): # train GAN for each condition
            
            data = data_c[c].to(device)

            discriminator.zero_grad()
            Dr_output = discriminator(data, c)
            errD_real = get_gan_loss(Dr_output,True)
            
            fake = generator(get_noise(data.shape[0]), c)
            Df_output = discriminator(fake.detach(), c)
            errD_fake = get_gan_loss(Df_output,False)

            errD = errD_real + errD_fake
                
            acc_r = Dr_output.mean().item() 
            acc_f = 1.0 - Df_output.mean().item() 
            acc = (acc_r + acc_f) / 2.0
            
            acc_list.append(acc) # calculate discriminator accuracy
            
            if (train_disc): # train discriminator if the last batch accuracy is less than 0.95
                errD.backward()
                optimizerD.step()

            generator.zero_grad() # train generator
            fake = generator(get_noise(), c)
            DGF_output = discriminator(fake, c)
            errG = get_gan_loss(DGF_output,True)
            errG.backward()
            optimizerG.step()
            
            d_loss.append(errD.mean().item())
            g_loss.append(errG.mean().item())

        generator.zero_grad() # train generator for combined loss
        discriminator.zero_grad()
        
        fix_noise = get_noise()

        fake0 = generator(fix_noise, 0) # generate for condition 0 and 1
        fake1 = generator(fix_noise, 1)
        
        fake1_rot = torch.rot90(fake1, 2, [1, 2]) # rotate condition 1
        fake_combined = (fake0 + fake1_rot) / 2.0 # combine them by averaging
         
        DGF_output_c = discriminator(fake_combined, 0) # train generator for combined output
        errG_c = get_gan_loss(DGF_output_c,True)
        errG_c.backward()
        optimizerG.step()

        train_disc = np.mean(acc_list) < 0.95 # decide for the next batch
    
    gen_out.append( fake0.detach().cpu() ) # return generated samples for condition 0, 1 and combined
    gen_out.append( fake1.detach().cpu() )
    gen_out.append( fake_combined.detach().cpu() )
    
    return np.mean(d_loss), np.mean(g_loss) , gen_out

utils.clear_folder(log_folder) # create log folder
log_file = open(log_folder +"logs.txt" ,"a") # open log file

d_list = []
g_list = []

pbar = tqdm( range(epoch_count+1) )
for i in pbar :
    
    startTime = time.time()
    
    d_loss, g_loss, gen = train_GAN_epoch() #train GAN for 1 epoch
    
    d_list.append(d_loss) # get discriminator and generator loss
    g_list.append(g_loss)
    
    utils.plot_graph([d_list,g_list], log_folder + "loss_graph") # plot loss graph up to that epoch

    epoch_time = time.time() - startTime
    
    writeString = "epoch %d --> d_loss:%0.3f g_loss:%0.3f time:%0.3f" % (i, d_loss, g_loss, epoch_time) # generate log string

    pbar.set_description(writeString)
    log_file.write(writeString + "\n") # write to log file
    log_file.flush()
    
    if(i%5 == 0): # save generated samples for each 5th epoch because it takes a long time to visualize the samples
        utils.visualize_all(gen, save=True, name = log_folder + "samples_epoch" + str(i))
        utils.save_checkpoints(cfg, os.path.join(log_folder + "checkpoints", 'ckpt-epoch-%04d.pth' % (i + 1)),
                                                 i + 1, generator, optimizerG, discriminator, optimizerD, best_iou, best_epoch)