
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms,utils,datasets
import matplotlib.pyplot as plt
import numpy as np

def use_gpu():
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available()
                                                         else torch.FloatTensor)
use_gpu()

latent_size = 64   ##Generator input embedding size
hidden_size = 256  ## FC layers number of neurons
image_size = 784   ## Images size (28x28)

num_epochs = 200
batch_size = 100

transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])

trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

class MNIST_Discriminator(nn.Module):
    def __init__(self):
        super(MNIST_Discriminator, self).__init__()

        self.discriminator=nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.discriminator(x)

class MNIST_Generator(nn.Module):
    def __init__(self):
        super(MNIST_Generator, self).__init__()

        self.generator = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh())

    def forward(self, x):
        return self.generator(x)


D = MNIST_Discriminator()#.to(device="cuda")
G = MNIST_Generator()#.to(device="cuda")


criterion = nn.BCELoss()  ##Binary cross entropy loss

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1)#.to(device='cuda')

        real_labels = torch.ones(batch_size, 1)#.to(device='cuda')
        fake_labels = torch.zeros(batch_size, 1)#.to(device='cuda')

        # Discriminator training

        # Compute BCE_Loss using real images
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        z = torch.randn(batch_size, latent_size)#.to(device='cuda')
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        d_loss.backward()
        d_optimizer.step()

        # Generator training
        z = torch.randn(batch_size, latent_size)#.to(device='cuda')
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(
                epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(),
                fake_score.mean().item()))

        if i + 1 == total_step and epoch % 10 == 0:
            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            grid_img = utils.make_grid(((fake_images.detach().cpu() + 1) / 2).clamp(0, 1), nrow=10)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.figure()

# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')

plt.show()