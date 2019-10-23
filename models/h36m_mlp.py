import torch
import torch.nn as nn

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class residual_linear(nn.Module):
    def __init__(self, nin, nout):
        super(residual_linear, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Linear(nin, nout),
            nn.ReLU(inplace=True)
        )
        self.long_path = nn.Sequential(
            nn.Linear(nin, nin//2),
            nn.ReLU(inplace=True),
            nn.Linear(nin//2, nin//2),
            nn.ReLU(inplace=True),
            nn.Linear(nin//2, nout),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(nout)
    
    def forward(self, input):
        return self.norm(self.shortcut(input) + self.long_path(input))


class encoder(nn.Module):
    def __init__(self, in_dim=17*3, out_dim=128, h_dim=128):
        super(encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim

        self.fc1 = residual_linear(in_dim, h_dim)
        self.fc2 = residual_linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, input):
        bs = input.shape[0]
        input = input.view(bs, -1)
        h1 = self.fc1(input)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        out = self.tanh(h3)
        #return out, 0
        return out, [h1, h2]


class decoder(nn.Module):
    def __init__(self, in_dim=128, out_dim=17*3, h_dim=128):
        super(decoder, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.fc1 = residual_linear(in_dim, h_dim)
        #self.fc2 = residual_linear(h_dim, h_dim)
        #self.fc3 = nn.Linear(h_dim, out_dim)
        self.fc2 = residual_linear(h_dim*2, h_dim)
        self.fc3 = nn.Linear(h_dim*2, out_dim)

    def forward(self, input):
        input, skip = input
        bs = input.shape[0]
        d1 = self.fc1(input)
        #d2 = self.fc2(d1)
        #out = self.fc3(d2)
        d2 = self.fc2(torch.cat([d1, skip[1]], 1))
        out = self.fc3(torch.cat([d2, skip[0]], 1))

        out = out.view(bs, 17, 3)
        return out


class encoder_old(nn.Module):
    def __init__(self, in_dim=17*3, out_dim=128, h_dim=128):
        super(encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, input):
        bs = input.shape[0]
        input = input.view(bs, -1)
        h1 = self.relu1(self.fc1(input))
        h2 = self.relu2(self.fc2(h1))
        h3 = self.fc3(h2)
        out = self.tanh(h3)

        return out, 0

class decoder_old(nn.Module):
    #def __init__(self, dim, nc=1):
    def __init__(self, in_dim=128, out_dim=17*3, h_dim=128):
        super(decoder, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        """
        input, _ = input
        bs = input.shape[0]
        d1 = self.relu1(self.fc1(input))
        d2 = self.relu2(self.fc2(d1))
        d3 = self.fc3(d2)
        out = self.sigmoid(d3)

        out = out.view(bs, 17, 3)
        return out



#class