from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

for epoch in range(10):
    writer.add_scalar("Y = 2x", 2*epoch, epoch)

writer.close()
