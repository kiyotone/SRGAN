import torch
from tqdm import tqdm
from loss import SRGANLosses  # Import the loss functions
import os


class SRGANTrainer:
    def __init__(self, version, generator, discriminator, gen_optimizer, disc_optimizer, device, vgg_loss_weight=1e-3):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        self.output_dir = f"out/{version}/models"
        os.makedirs(self.output_dir, exist_ok=True)
        self.latest_epoch = 0
        
    
        # Initialize loss functions
        self.losses = SRGANLosses(device, vgg_loss_weight)

    def train(self, train_loader, num_epochs=10):
        gen_losses, disc_losses = [], []
        
        start_epoch = self.latest_epoch        
        end_epoch = start_epoch + num_epochs

        for epoch in range(start_epoch , end_epoch):
            gen_loss_total = 0.0
            disc_loss_total = 0.0
            

            progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}")

            for batch in progress_bar:
                low_res, high_res = batch
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                
                # Train the discriminator
                fake = self.generator(low_res)
                disc_real = self.discriminator(high_res)
                disc_fake = self.discriminator(fake.detach())
                
                loss_disc = self.losses.discriminator_loss(disc_real, disc_fake)
                self.disc_optimizer.zero_grad()
                loss_disc.backward()
                self.disc_optimizer.step()
                
                # Train the generator
                disc_fake = self.discriminator(fake)
                gen_loss = self.losses.generator_loss(fake, high_res, disc_fake)
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()
                
                gen_loss_total += gen_loss.item()
                disc_loss_total += loss_disc.item()
                gen_losses.append(gen_loss.item())
                disc_losses.append(loss_disc.item())
                
                progress_bar.set_postfix(gen_loss=gen_loss.item(), disc_loss=loss_disc.item())
            
            # Save models
            torch.save(self.generator.state_dict(), f"{self.output_dir}/generator_epoch_{epoch+1}.pth")
            torch.save(self.discriminator.state_dict(), f"{self.output_dir}/discriminator_epoch_{epoch+1}.pth")
            
            gen_loss_total /= len(train_loader)
            disc_loss_total /= len(train_loader)
            print(f"Epoch {epoch+1}: Generator loss: {gen_loss_total}, Discriminator loss: {disc_loss_total}")
            
        # Log the losses
        with open(f"{self.output_dir}/losses.txt", "w") as f:
            f.write("Generator Losses\n")
            for loss in gen_losses:
                f.write(f"{loss}\n")
            f.write("\nDiscriminator Losses\n")
            for loss in disc_losses:
                f.write(f"{loss}\n")


    def get_losses(self):
        with open(f"{self.output_dir}/losses.txt", "r") as f:
            lines = f.readlines()
            gen_losses = [float(loss) for loss in lines[1:lines.index("\n")]]
            disc_losses = [float(loss) for loss in lines[lines.index("\n")+1:]]
        return gen_losses, disc_losses
    

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model
    
    def load_generator(self , epoch):
        return self.load_model(self.generator, f"{self.output_dir}/generator_epoch_{epoch}.pth")
    
    def load_discriminator(self , epoch):
        return self.load_model(self.discriminator, f"{self.output_dir}/discriminator_epoch_{epoch}.pth")
    
    def load_latest_models(self):
        gen_epoch = max([int(file.split("_")[-1].split(".")[0]) for file in os.listdir(self.output_dir) if "generator" in file])
        disc_epoch = max([int(file.split("_")[-1].split(".")[0]) for file in os.listdir(self.output_dir) if "discriminator" in file])
        print(f"Loading generator model from epoch {gen_epoch}")
        print(f"Loading discriminator model from epoch {disc_epoch}")
        self.load_generator(gen_epoch)
        self.load_discriminator(disc_epoch)
        self.latest_epoch = max(gen_epoch, disc_epoch)
        return gen_epoch, disc_epoch
        