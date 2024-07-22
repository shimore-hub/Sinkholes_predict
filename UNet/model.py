import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        # print the kernels
        # print(self.conv1.weight.size())
        # print(self.conv1.weight)
        num_kernels = self.conv1.out_channels
        # print(f"Number of kernels in the convolutional layer: {num_kernels}")
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        # print the kernels
        # print(self.conv2.weight.size())
        # print(self.conv2.weight)
        num_kernels2 = self.conv2.out_channels
        # print(f"Number of kernels in the convolutional layer: {num_kernels2}")
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))  # Adjust the stride to maintain spatial dimensions

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        # print(type(inputs), type(skip))
        # print(inputs.shape, skip.shape)
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        # print(f's1 shape: {s1.shape}, p1 shape: {p1.shape}')
        s2, p2 = self.e2(p1)
        # print(f's2 shape: {s2.shape}, p2 shape: {p2.shape}')
        s3, p3 = self.e3(p2)
        # print(f's3 shape: {s3.shape}, p3 shape: {p3.shape}')
        s4, p4 = self.e4(p3)
        # print(f's4 shape: {s4.shape}, p4 shape: {p4.shape}')

        """ Bottleneck """

        # print("s4 shape:", s4.shape)
        # print("p4 shape:", p4.shape)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # print("d1 shape:", d1.shape)
        # print("d2 shape:", d2.shape)
        # print("d3 shape:", d3.shape)
        # print("d4 shape:", d4.shape)

        outputs = self.outputs(d4)

        return outputs


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss


if __name__ == "__main__":
    x = torch.randn((2, 1, 200, 200))
    f = build_unet()
    y = f(x)
    print(y.shape)
