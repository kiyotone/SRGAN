import torch
import matplotlib.pyplot as plt
from models.generator import Generator

def generate_and_plot(generator, num_samples=5, latent_dim=100, seq_length=30):
    """
    Generate random sequences using the trained Generator and plot them.
    """
    # Generate random latent vectors
    z = torch.randn(num_samples, latent_dim)
    with torch.no_grad():
        fake_sequences = generator(z)

    # Plot the generated sequences
    for i in range(num_samples):
        plt.plot(fake_sequences[i].cpu().numpy())
        plt.title(f"Generated Sequence {i+1}")
        plt.show()

def plot_training_loss(d_losses, g_losses):
    """
    Plot the discriminator and generator loss curves during training.
    """
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.show()

# Example usage (this will be used after training is completed):
if __name__ == "__main__":
    # Example: Load the trained generator
    generator = Generator(seq_length=30, feature_dim=10, latent_dim=100)
    generator.load_state_dict(torch.load('models/generator.pth'))

    # Visualize generated sequences
    generate_and_plot(generator, num_samples=5)

    # Example: Plot training loss curves (after training)
    # plot_training_loss(d_losses, g_losses)  # Uncomment this after training
