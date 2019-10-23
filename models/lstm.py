import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        #self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        self.hidden = hidden
        return hidden

    def init_hidden_(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        self.hidden = hidden
        #return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        self.hidden = hidden
        return hidden

    def init_hidden_(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        self.hidden = hidden
        #return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        #return eps.add_(mu)
        #return eps.mul(logvar)
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        #import pdb
        #pdb.set_trace()
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
            

class gaussian_bilstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.fw_lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.bw_lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.fw_hidden = self.init_hidden()
        self.bw_hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def init_hidden_(self):
        fw_hidden = []
        bw_hidden = []
        for i in range(self.n_layers):
            fw_hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                              Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
            bw_hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                              Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))

        self.fw_hidden = fw_hidden
        self.bw_hidden = bw_hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def one_step(self, input, direction="forward"):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded

        for i in range(self.n_layers):
            if direction == "forward":
                self.fw_hidden[i] = self.fw_lstm[i](h_in, self.fw_hidden[i])
                h_in = self.fw_hidden[i][0]
            else:
                self.bw_hidden[i] = self.bw_lstm[i](h_in, self.bw_hidden[i])
                h_in = self.bw_hidden[i][0]

        return h_in

    def forward(self, input):
        fw_h_in = self.one_step(input, "forward")
        bw_h_in = self.one_step(input, "forward")

        h_in = torch.cat([fw_h_in, bw_h_in], 1)

        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar