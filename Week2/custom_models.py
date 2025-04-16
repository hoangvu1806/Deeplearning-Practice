import torch
import torch.nn as nn
import torch.nn.init as init

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.W_ih = nn.ParameterList([nn.Parameter(torch.empty(embedding_dim if i == 0 else hidden_size, hidden_size)) 
                                      for i in range(num_layers)])
        self.W_hh = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, hidden_size)) 
                                      for _ in range(num_layers)])
        self.b_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) 
                                     for _ in range(num_layers)])
        
        self.W_ho = nn.Parameter(torch.empty(hidden_size, output_size))
        self.b_o = nn.Parameter(torch.zeros(output_size))
        
        # Khởi tạo Xavier cho tất cả trọng số
        for w in self.W_ih:
            init.xavier_uniform_(w)
        for w in self.W_hh:
            init.xavier_uniform_(w)
        init.xavier_uniform_(self.W_ho)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            for l in range(self.num_layers):
                if l == 0:
                    h[l] = torch.tanh(x_t @ self.W_ih[l] + h[l] @ self.W_hh[l] + self.b_h[l])
                else:
                    h[l] = torch.tanh(h[l-1] @ self.W_ih[l] + h[l] @ self.W_hh[l] + self.b_h[l])
        
        logits = h[-1] @ self.W_ho + self.b_o
        return logits

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.W_ii = nn.ParameterList([nn.Parameter(torch.empty(embedding_dim if i == 0 else hidden_size, hidden_size)) 
                                      for i in range(num_layers)])
        self.W_hi = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, hidden_size)) 
                                      for _ in range(num_layers)])
        self.b_i = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) 
                                     for _ in range(num_layers)])
        
        self.W_if = nn.ParameterList([nn.Parameter(torch.empty(embedding_dim if i == 0 else hidden_size, hidden_size)) 
                                      for i in range(num_layers)])
        self.W_hf = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, hidden_size)) 
                                      for _ in range(num_layers)])
        self.b_f = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) 
                                     for _ in range(num_layers)])
        
        self.W_ig = nn.ParameterList([nn.Parameter(torch.empty(embedding_dim if i == 0 else hidden_size, hidden_size)) 
                                      for i in range(num_layers)])
        self.W_hg = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, hidden_size)) 
                                      for _ in range(num_layers)])
        self.b_g = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) 
                                     for _ in range(num_layers)])
        
        self.W_io = nn.ParameterList([nn.Parameter(torch.empty(embedding_dim if i == 0 else hidden_size, hidden_size)) 
                                      for i in range(num_layers)])
        self.W_ho = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, hidden_size)) 
                                      for _ in range(num_layers)])
        self.b_o = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) 
                                     for _ in range(num_layers)])
        
        self.W_out = nn.Parameter(torch.empty(hidden_size, output_size))
        self.b_out = nn.Parameter(torch.zeros(output_size))
        
        # Khởi tạo Xavier cho tất cả trọng số
        for param_list in [self.W_ii, self.W_hi, self.W_if, self.W_hf, self.W_ig, self.W_hg, self.W_io, self.W_ho]:
            for w in param_list:
                init.xavier_uniform_(w)
        init.xavier_uniform_(self.W_out)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        
        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            for l in range(self.num_layers):
                if l == 0:
                    i = torch.sigmoid(x_t @ self.W_ii[l] + h[l] @ self.W_hi[l] + self.b_i[l])
                    f = torch.sigmoid(x_t @ self.W_if[l] + h[l] @ self.W_hf[l] + self.b_f[l])
                    g = torch.tanh(x_t @ self.W_ig[l] + h[l] @ self.W_hg[l] + self.b_g[l])
                    o = torch.sigmoid(x_t @ self.W_io[l] + h[l] @ self.W_ho[l] + self.b_o[l])
                else:
                    i = torch.sigmoid(h[l-1] @ self.W_ii[l] + h[l] @ self.W_hi[l] + self.b_i[l])
                    f = torch.sigmoid(h[l-1] @ self.W_if[l] + h[l] @ self.W_hf[l] + self.b_f[l])
                    g = torch.tanh(h[l-1] @ self.W_ig[l] + h[l] @ self.W_hg[l] + self.b_g[l])
                    o = torch.sigmoid(h[l-1] @ self.W_io[l] + h[l] @ self.W_ho[l] + self.b_o[l])
                c[l] = f * c[l] + i * g
                h[l] = o * torch.tanh(c[l])
        
        logits = h[-1] @ self.W_out + self.b_out
        return logits