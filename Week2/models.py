import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        output_size,
        pad_idx,
        dropout=0.3,
    ):
        """
        Args:
            vocab_size (int): Kích thước từ vựng.
            embedding_dim (int): Kích thước vector embedding.
            hidden_size (int): Số nơron trong lớp ẩn.
            num_layers (int): Số lớp RNN.
            output_size (int): Số lớp đầu ra (2 cho phân loại nhị phân).
            pad_idx (int): Chỉ số của token <pad>.
            dropout (float): Tỷ lệ dropout (mặc định 0.3).
        """
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.pad_idx = pad_idx

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor đầu vào, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits đầu ra, shape (batch_size, output_size).
        """
        lengths = (x != self.pad_idx).sum(dim=1).cpu()  # Tính độ dài thực tế
        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.rnn(packed)
        logits = self.fc(self.dropout(hidden[-1]))
        return logits


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        output_size,
        pad_idx,
        dropout=0.3,
    ):
        """
        Mô hình LSTM sử dụng nn.LSTM.

        Args:
            vocab_size (int): Kích thước từ vựng.
            embedding_dim (int): Kích thước vector embedding.
            hidden_size (int): Số nơron trong lớp ẩn.
            num_layers (int): Số lớp LSTM.
            output_size (int): Số lớp đầu ra (2 cho phân loại nhị phân).
            pad_idx (int): Chỉ số của token <pad>.
            dropout (float): Tỷ lệ dropout (mặc định 0.3).
        """
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.pad_idx = pad_idx

    def forward(self, x):
        """
        Lan truyền tiến.

        Args:
            x (torch.Tensor): Tensor đầu vào, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits đầu ra, shape (batch_size, output_size).
        """
        lengths = (x != self.pad_idx).sum(dim=1).cpu()  # Tính độ dài thực tế
        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed)
        logits = self.fc(self.dropout(hidden[-1]))
        return logits
