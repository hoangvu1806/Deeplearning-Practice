import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Lớp Attention cơ bản (Bahdanau style)."""
    def __init__(self, hidden_dim: int, attention_dim: int):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        
        # Lớp attention
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        
    def forward(self, features):
        """
        Args:
            features (torch.Tensor): Output từ LSTM. Shape: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context_vector (torch.Tensor): Vector ngữ cảnh sau khi áp dụng attention.
                                          Shape: (batch_size, hidden_dim)
            attention_weights (torch.Tensor): Trọng số attention. 
                                              Shape: (batch_size, seq_len, 1)
        """
        # (batch_size, seq_len, attention_dim)
        attention = torch.tanh(self.W(features))
        
        # (batch_size, seq_len, 1)
        attention_weights = F.softmax(self.V(attention), dim=1)
        
        # (batch_size, hidden_dim)
        context_vector = torch.sum(attention_weights * features, dim=1)
        
        return context_vector, attention_weights

class LSTMAttention(nn.Module):
    """Mô hình LSTM hai chiều kết hợp với Attention để phân loại văn bản."""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                attention_dim: int, output_dim: int, num_layers: int, 
                bidirectional: bool, dropout: float, pad_idx: int):
        super(LSTMAttention, self).__init__()
        
        # Định nghĩa các tham số
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Lớp embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Lớp LSTM
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        
        # Lớp Attention
        self.attention = Attention(hidden_dim * self.num_directions, attention_dim)
        
        # Lớp dropout
        self.dropout = nn.Dropout(dropout)
        
        # Lớp fully connected
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        
    def forward(self, text):
        """
        Args:
            text (torch.Tensor): Input tensor chứa indices của từ. Shape: (batch_size, seq_len)
        
        Returns:
            prediction (torch.Tensor): Logits đầu ra của mô hình. Shape: (batch_size, output_dim)
            attn_weights (torch.Tensor): Trọng số attention. Shape: (batch_size, seq_len, 1)
        """
        # Nhúng từ vựng
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)
        
        # Đưa qua LSTM
        # output: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        # cell: (num_layers * num_directions, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Áp dụng attention layer
        # context: (batch_size, hidden_dim * num_directions)
        # attn_weights: (batch_size, seq_len, 1)
        context, attn_weights = self.attention(output)
        
        # Áp dụng dropout
        context = self.dropout(context)
        
        # Phân loại
        # prediction: (batch_size, output_dim)
        prediction = self.fc(context)
        
        return prediction, attn_weights 