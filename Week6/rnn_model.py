import os
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math

# Import configurations
from config import *

# Set seed for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Positional Encoding class - helps model understand token positions
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# Custom dataset for translation tasks
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.source = data['source']
        self.target = data['target']
        assert len(self.source) == len(self.target), "Source and target must have the same length"
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        return {
            'source': np.array(self.source[idx], dtype=np.int64),
            'target': np.array(self.target[idx], dtype=np.int64)
        }

# Collate function for DataLoader
def collate_fn(batch):
    sources = [torch.from_numpy(item['source']) for item in batch]
    targets = [torch.from_numpy(item['target']) for item in batch]
    
    # Pad sequences
    sources_padded = pad_sequence(sources, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return {'source': sources_padded, 'target': targets_padded}

# Encoder class with Bidirectional GRU
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_directions = 2  # Bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)  # Reduce bidirectional output dimension
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hid_dim)  # Layer normalization for better training
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]
        
        # Pass through bidirectional GRU
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim * 2]  # *2 because bidirectional
        # hidden: [n_layers * 2, batch_size, hid_dim]  # *2 because bidirectional
        
        # Combine forward and backward states for each layer
        hidden_combined = []
        for i in range(self.n_layers):
            # Concatenate forward and backward hidden states
            h_forward = hidden[2*i]
            h_backward = hidden[2*i + 1]
            h_combined = torch.cat((h_forward, h_backward), dim=1)
            
            # Project to original hidden dimension
            h_combined = self.fc(h_combined)
            h_combined = self.layer_norm(h_combined)  # Apply layer normalization
            hidden_combined.append(h_combined)
        
        # Stack the layers back together
        hidden = torch.stack(hidden_combined)
        # hidden: [n_layers, batch_size, hid_dim]
        
        # Project outputs to original hidden dimension
        outputs = self.fc(outputs)
        outputs = self.layer_norm(outputs)  # Apply layer normalization
        # outputs: [batch_size, src_len, hid_dim]
        
        return outputs, hidden

# Enhanced Attention layer with Scaled Dot-Product Attention
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # Multi-head attention components
        self.query_proj = nn.Linear(hid_dim, hid_dim)
        self.key_proj = nn.Linear(hid_dim, hid_dim)
        self.value_proj = nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
        # Output projection
        self.fc_out = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hid_dim)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Transform inputs to queries, keys, and values
        queries = self.query_proj(hidden.unsqueeze(1))  # [batch_size, 1, hid_dim]
        keys = self.key_proj(encoder_outputs)  # [batch_size, src_len, hid_dim]
        values = self.value_proj(encoder_outputs)  # [batch_size, src_len, hid_dim]
        
        # Scaled dot-product attention
        # queries * keys^T / sqrt(hid_dim)
        energy = torch.bmm(queries, keys.transpose(1, 2)) / self.scale  # [batch_size, 1, src_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(energy, dim=2)  # [batch_size, 1, src_len]
        attention_weights = self.dropout(attention_weights)  # Apply dropout for regularization
        
        # Weighted sum of values
        context = torch.bmm(attention_weights, values)  # [batch_size, 1, hid_dim]
        context = context.squeeze(1)  # [batch_size, hid_dim]
        
        # Concatenate context with hidden for richer representation
        combined = torch.cat((context, hidden), dim=1)  # [batch_size, hid_dim*2]
        output = self.fc_out(combined)  # [batch_size, hid_dim]
        output = self.layer_norm(output + hidden)  # Residual connection and normalization
        
        return attention_weights.squeeze(1), output

# Enhanced Decoder class with improved attention and residual connections
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # Embedding with positional encoding
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len=100)
        
        # Attention mechanism
        self.attention = Attention(hid_dim)
        
        # GRU layers with residual connections
        self.rnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Input to first layer has different dimensions
        self.rnn_layers.append(nn.GRU(emb_dim + hid_dim, hid_dim, 1, batch_first=True))
        self.layer_norms.append(nn.LayerNorm(hid_dim))
        
        # Additional layers
        for _ in range(1, n_layers):
            self.rnn_layers.append(nn.GRU(hid_dim, hid_dim, 1, batch_first=True))
            self.layer_norms.append(nn.LayerNorm(hid_dim))
        
        # Output projection with gating mechanism
        self.gate = nn.Linear(hid_dim * 2 + emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        
        # Embed input tokens and add positional encoding
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]
        embedded = self.pos_encoder(embedded)  # Add positional encoding
        embedded = self.dropout(embedded)  # Apply dropout
        
        # Calculate attention weights and context using the top layer hidden state
        attn_weights, context_vector = self.attention(hidden[-1], encoder_outputs)
        # attn_weights: [batch_size, src_len]
        # context_vector: [batch_size, hid_dim]
        
        # Prepare decoder input (embedded + context)
        rnn_input = torch.cat((embedded, context_vector.unsqueeze(1)), dim=2)  # [batch_size, 1, emb_dim + hid_dim]
        
        # Process through RNN layers with residual connections
        layer_outputs = []
        layer_input = rnn_input
        
        for i in range(self.n_layers):
            # Current layer's hidden state
            layer_hidden = hidden[i].unsqueeze(0)  # [1, batch_size, hid_dim]
            
            # Run through GRU
            if i == 0:
                # First layer takes the concatenated input
                output, new_hidden = self.rnn_layers[i](layer_input, layer_hidden)
            else:
                # Subsequent layers take previous layer output
                output, new_hidden = self.rnn_layers[i](layer_input, layer_hidden)
            
            # Apply layer normalization and residual connection if not first layer
            if i > 0:
                output = self.layer_norms[i](output + layer_input)  # Residual connection
            else:
                output = self.layer_norms[i](output)  # No residual for first layer with different dimensions
            
            # Update layer input for next layer
            layer_input = output
            layer_outputs.append(new_hidden.squeeze(0))  # Store new hidden state
        
        # Update hidden states for all layers
        hidden = torch.stack(layer_outputs)  # [n_layers, batch_size, hid_dim]
        
        # Prepare for output projection
        output = output.squeeze(1)  # [batch_size, hid_dim]
        embedded = embedded.squeeze(1)  # [batch_size, emb_dim]
        
        # Gating mechanism to control information flow
        gate_input = torch.cat((output, context_vector, embedded), dim=1)  # [batch_size, hid_dim*2 + emb_dim]
        gate = torch.sigmoid(self.gate(gate_input))  # [batch_size, hid_dim]
        
        # Apply gate and project to vocabulary
        gated_output = gate * output + (1 - gate) * context_vector  # Weighted combination
        prediction = self.fc_out(gated_output)  # [batch_size, output_dim]
        
        return prediction, hidden

# Enhanced Seq2Seq model with improved encoder and decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Add a bridge layer between encoder and decoder
        self.bridge = nn.Linear(encoder.hid_dim, decoder.hid_dim)
        self.layer_norm = nn.LayerNorm(decoder.hid_dim)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.8):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        # Using a higher teacher_forcing_ratio (0.8 instead of 0.5) to help model learn better
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Initialize decoder hidden states (bridge between encoder and decoder)
        hidden = []
        for i in range(self.decoder.n_layers):
            # Apply bridge transformation to each encoder hidden state
            h = self.bridge(encoder_hidden[i])
            h = self.layer_norm(h)  # Apply layer normalization
            hidden.append(h)
        hidden = torch.stack(hidden)  # [n_layers, batch_size, hid_dim]
        
        # First input to the decoder is the <sos> token
        input = trg[:, 0]
        
        # Schedule sampling strategy - gradually reduce teacher forcing
        tf_ratio = teacher_forcing_ratio
        
        # Scheduled sampling - decrease teacher forcing as training progresses
        # This implementation uses linear decay
        tf_ratio_step = tf_ratio / trg_len  # Linear decay
        
        for t in range(1, trg_len):
            # Decode
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide if we're going to use teacher forcing (with decaying probability)
            teacher_force = random.random() < tf_ratio
            tf_ratio = max(0.1, tf_ratio - tf_ratio_step)  # Decay but keep minimum at 0.1
            
            # Top-k sampling for better exploration when not using teacher forcing
            if not teacher_force:
                # Top-k sampling with k=3
                topk_probs, topk_idx = torch.topk(F.softmax(output, dim=1), k=3)
                sampled_idx = torch.multinomial(topk_probs, 1).squeeze(1)
                top1 = torch.gather(topk_idx, 1, sampled_idx.unsqueeze(1)).squeeze(1)
            else:
                # Use ground truth token
                top1 = trg[:, t]
            
            # Update input for next step
            input = top1
        
        return outputs

# Function to initialize model based on config
def init_model(config, src_vocab_size, trg_vocab_size):
    # Fix dropout for single layer models
    if config['num_layers'] == 1 and config['dropout'] > 0:
        print(f"Setting dropout to 0.0 for single layer model in {config['name']}")
        config['dropout'] = 0.0
    
    # Initialize encoder and decoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=config['embedding_size'],
        hid_dim=config['hidden_size'],
        n_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    decoder = Decoder(
        output_dim=trg_vocab_size,
        emb_dim=config['embedding_size'],
        hid_dim=config['hidden_size'],
        n_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Create Seq2Seq model
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Initialize weights
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                # Use Xavier/Glorot initialization for better convergence
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.uniform_(param.data, -0.08, 0.08)
            elif 'bias' in name:
                nn.init.zeros_(param.data)  # Initialize biases to zero
    
    model.apply(init_weights)
    
    # Setup optimizer
    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    return model, optimizer

# Function to calculate loss
def calculate_loss(output, target, criterion, pad_idx=0):
    # output: [batch_size, trg_len, output_dim]
    # target: [batch_size, trg_len]
    
    # Reshape output and target for loss calculation
    output_dim = output.shape[-1]
    output = output[:, 1:].reshape(-1, output_dim)  # Exclude first token (SOS)
    target = target[:, 1:].reshape(-1)  # Exclude first token (SOS)
    
    # Calculate loss, ignoring padding tokens (handled by criterion with ignore_index=pad_idx)
    loss = criterion(output, target)
    
    return loss

# Function to train the model for one epoch
def train_epoch(model, data_loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc="Training")
    
    for i, batch in enumerate(progress_bar):
        src = batch['source'].to(device)
        trg = batch['target'].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        loss = calculate_loss(output, trg, criterion)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Removed saving checkpoint every 500 batches for efficiency
    
    return epoch_loss / len(data_loader)

# Function to evaluate the model
def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            output = model(src, trg, teacher_forcing_ratio=0)  # No teacher forcing during evaluation
            
            loss = calculate_loss(output, trg, criterion)
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)

# Function to translate a sentence with the enhanced model architecture
def translate_sentence(model, sentence, src_vocab, trg_vocab, device, max_len=16):
    model.eval()
    
    # Convert sentence to indices
    if isinstance(sentence, str):
        tokens = sentence.lower().split()
        sentence = [src_vocab.get(token, src_vocab[UNK_TOKEN]) for token in tokens]
    
    # Add SOS and EOS tokens
    sentence = [src_vocab[SOS_TOKEN]] + sentence + [src_vocab[EOS_TOKEN]]
    src_tensor = torch.LongTensor(sentence).unsqueeze(0).to(device)
    
    # Encode the sentence
    with torch.no_grad():
        encoder_outputs, encoder_hidden = model.encoder(src_tensor)
        
        # Initialize decoder hidden states using bridge layer
        hidden = []
        for i in range(model.decoder.n_layers):
            # Apply bridge transformation to each encoder hidden state
            h = model.bridge(encoder_hidden[i])
            h = model.layer_norm(h)  # Apply layer normalization
            hidden.append(h)
        hidden = torch.stack(hidden)  # [n_layers, batch_size, hid_dim]
    
    # Start with SOS token for decoding
    trg_idx = [trg_vocab[SOS_TOKEN]]
    
    # Start decoding - limit to max_len (16) tokens
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        # Get top-k sampling (k=5) for more diverse outputs
        topk_probs, topk_idx = torch.topk(F.softmax(output, dim=1), k=2)
        
        # Apply temperature to sharpen/soften the distribution
        temperature = 1.0  # <1 = sharper, >1 = softer
        scaled_probs = topk_probs ** (1/temperature)
        scaled_probs = scaled_probs / scaled_probs.sum()  # Re-normalize
        
        # Get predicted token with some randomness for diversity
        if random.random() < 0.9:  # 90% of the time, use sampling
            # Sample from top-k based on probability
            sampled_idx = torch.multinomial(scaled_probs, 1).squeeze()
            pred_token = topk_idx[0, sampled_idx].item()
            
            # Check for repetition and avoid if possible
            repeat_count = 0
            for i in range(1, min(3, len(trg_idx))):
                if pred_token == trg_idx[-i]:
                    repeat_count += 1
            
            # If excessive repetition, try another token
            if repeat_count >= 2 and len(scaled_probs) > 1:
                # Remove the repeated token from consideration
                mask = torch.ones_like(scaled_probs)
                mask[sampled_idx] = 0
                masked_probs = scaled_probs * mask
                if masked_probs.sum() > 0:  # Check if we have valid alternatives
                    masked_probs = masked_probs / masked_probs.sum()  # Re-normalize
                    new_idx = torch.multinomial(masked_probs, 1).squeeze()
                    pred_token = topk_idx[0, new_idx].item()
        else:
            # 10% of the time, use greedy selection
            pred_token = topk_idx[0, 0].item()
        
        trg_idx.append(pred_token)
        
        # Stop if EOS token is predicted or if we've reached max length
        if pred_token == trg_vocab[EOS_TOKEN] or len(trg_idx) >= max_len + 1:  # +1 for SOS
            break
    
    # Convert indices to tokens (exclude SOS and EOS)
    trg_tokens = []
    for idx in trg_idx[1:-1]:  # Bỏ qua SOS và EOS
        # Tìm token tương ứng với index, bỏ qua các token đặc biệt nếu cần
        if idx in trg_vocab.values():
            token = list(trg_vocab.keys())[list(trg_vocab.values()).index(idx)]
            # Có thể loại bỏ các token đặc biệt không cần thiết trong kết quả nếu muốn
            # Ở đây chỉ loại bỏ PAD_TOKEN nếu vô tình xuất hiện
            if token != PAD_TOKEN:
                trg_tokens.append(token)
        else:
            # Xử lý nếu index không tồn tại trong từ điển
            trg_tokens.append(UNK_TOKEN)
    
    return trg_tokens

# Function to calculate BLEU score
def calculate_bleu(model, data_loader, src_vocab, trg_vocab, device, n_examples=100):
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= n_examples:
                break
                
            src = batch['source'][0].numpy().tolist()  # Take only one example from the batch
            trg = batch['target'][0].numpy().tolist()  # Take only one example from the batch
            
            # Translate source sentence
            translation = translate_sentence(model, src, src_vocab, trg_vocab, device)
            
            # Convert target sentence to tokens (excluding SOS and EOS)
            trg_tokens = []
            for idx in trg[1:-1]:  # Bỏ qua SOS và EOS
                if idx in trg_vocab.values():
                    token = list(trg_vocab.keys())[list(trg_vocab.values()).index(idx)]
                    if token != PAD_TOKEN:  # Bỏ qua padding token
                        trg_tokens.append(token)
                else:
                    trg_tokens.append(UNK_TOKEN)
            
            # Add to lists
            references.append([trg_tokens])
            hypotheses.append(translation)
    
    # Calculate BLEU score with smoothing
    smooth = SmoothingFunction().method1  # Smooth by adding 1 to both numerator and denominator (method1)
    return corpus_bleu(references, hypotheses, smoothing_function=smooth) * 100

# Function to save the metrics plot
def save_metrics_plot(train_losses, val_losses, bleu_scores, config_name):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss - {config_name}')
    
    # Plot BLEU scores
    plt.subplot(1, 2, 2)
    plt.plot(bleu_scores, 'g-', label='BLEU Score')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title(f'BLEU Score - {config_name}')
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/{config_name}_metrics.png")
    plt.close()

# Function to train a model with a specific configuration
def train_model(config, train_data, val_data, test_data, src_vocab, trg_vocab):
    print(f"\nTraining model with configuration: {config['name']}")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize model
    model, optimizer = init_model(config, len(src_vocab), len(trg_vocab))
    
    # Đảm bảo sử dụng đúng PAD index (thường là 0) để bỏ qua khi tính loss
    pad_idx = src_vocab[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # Ignore padding index
    
    # Create data loaders
    train_dataset = TranslationDataset(train_data)
    val_dataset = TranslationDataset(val_data)
    test_dataset = TranslationDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # Batch size 1 for BLEU calculation
    
    # Variables to track training progress
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    bleu_scores = []
    train_times = []
    
    # Start training
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['clip'], device)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Calculate BLEU score on a subset of test data
        bleu_score = calculate_bleu(model, test_loader, src_vocab, trg_vocab, device, n_examples=10)
        bleu_scores.append(bleu_score)
        
        # Calculate epoch time
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        train_times.append(end_time - start_time)
        
        # Update best validation loss if applicable
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        print(f"Epoch: {epoch+1}/{config['num_epochs']} | Time: {epoch_mins}m {epoch_secs:.2f}s")
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | BLEU Score: {bleu_score:.5f}")
        
        # Hiển thị một mẫu dịch thử
        if len(val_data['source']) > 0:
            # Chọn ngẫu nhiên một mẫu từ tập validation
            idx = random.randint(0, len(val_data['source']) - 1)
            src_sample = val_data['source'][idx]
            trg_sample = val_data['target'][idx]
            
            # Dịch mẫu
            translation = translate_sentence(model, src_sample, src_vocab, trg_vocab, device)
            
            # Hiển thị kết quả
            src_tokens = [list(src_vocab.keys())[list(src_vocab.values()).index(idx)] for idx in src_sample 
                        if idx != src_vocab[SOS_TOKEN] and idx != src_vocab[EOS_TOKEN] and idx != src_vocab[PAD_TOKEN]]
            trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(idx)] for idx in trg_sample
                        if idx != trg_vocab[SOS_TOKEN] and idx != trg_vocab[EOS_TOKEN] and idx != trg_vocab[PAD_TOKEN]]
            
            print("\nMẫu dịch thử:")
            print(f"Tiếng Anh: {' '.join(src_tokens)}")
            print(f"Tiếng Việt chuẩn: {' '.join(trg_tokens)}")
            print(f"Dịch của mô hình: {' '.join(translation)}\n")
    
    # Save metrics plot
    save_metrics_plot(train_losses, val_losses, bleu_scores, config['name'])
    
    # Save all metrics as numpy arrays
    np.savez(f"results/{config['name']}_metrics.npz", 
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             bleu_scores=np.array(bleu_scores),
             train_times=np.array(train_times))
    
    # Calculate final BLEU score on the full test set
    final_bleu = calculate_bleu(model, test_loader, src_vocab, trg_vocab, device, n_examples=100)
    
    # Save final checkpoint
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'src_vocab_size': len(src_vocab),
        'trg_vocab_size': len(trg_vocab),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'final_bleu_score': final_bleu
    }
    torch.save(final_checkpoint, f"checkpoints/{config['name']}_final.pt")
    
    # Return results
    return {
        'name': config['name'],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_bleu': final_bleu,
        'best_val_loss': best_val_loss,
        'avg_epoch_time': sum(train_times) / len(train_times)
    }

def main():
    print("Loading data...")
    
    # Load vocabularies
    with open(VOCAB_EN_PATH, 'rb') as f:
        src_vocab = pickle.load(f)
    
    with open(VOCAB_VI_PATH, 'rb') as f:
        trg_vocab = pickle.load(f)
    
    # Load training, validation, and test data
    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(VAL_DATA_PATH, 'rb') as f:
        val_data = pickle.load(f)
    
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)
        
    # Convert tuple data to dictionary format if needed
    if isinstance(train_data, tuple) and len(train_data) == 2:
        train_data = {'source': train_data[0], 'target': train_data[1]}
    if isinstance(val_data, tuple) and len(val_data) == 2:
        val_data = {'source': val_data[0], 'target': val_data[1]}
    if isinstance(test_data, tuple) and len(test_data) == 2:
        test_data = {'source': test_data[0], 'target': test_data[1]}
    
    print(f"Loaded vocabularies: English: {len(src_vocab)}, Vietnamese: {len(trg_vocab)}")
    print(f"Loaded datasets: Train: {len(train_data['source'])}, Val: {len(val_data['source'])}, Test: {len(test_data['source'])}")
    
    # Train models with all configurations
    results = []
    for config in RNN_CONFIGS:
        result = train_model(config, train_data, val_data, test_data, src_vocab, trg_vocab)
        results.append(result)
    
    # Calculate overall metrics
    print("\nOverall Results:")
    print("=================\n")
    
    train_losses = [result['final_train_loss'] for result in results]
    val_losses = [result['final_val_loss'] for result in results]
    bleu_scores = [result['final_bleu'] for result in results]
    epoch_times = [result['avg_epoch_time'] for result in results]
    
    print(f"Mean Training Loss: {mean(train_losses):.4f} ± {stdev(train_losses):.4f}")
    print(f"Mean Validation Loss: {mean(val_losses):.4f} ± {stdev(val_losses):.4f}")
    print(f"Mean BLEU Score: {mean(bleu_scores):.2f} ± {stdev(bleu_scores):.2f}")
    print(f"Mean Epoch Time: {mean(epoch_times):.2f}s ± {stdev(epoch_times):.2f}s")
    
    # Find best model based on BLEU score
    best_model = max(results, key=lambda x: x['final_bleu'])
    print(f"\nBest Model: {best_model['name']} with BLEU Score: {best_model['final_bleu']:.2f}")
    
    # Save overall results
    with open('results/rnn_overall_results.txt', 'w') as f:
        f.write("RNN Model Results\n")
        f.write("================\n\n")
        
        for result in results:
            f.write(f"Config: {result['name']}\n")
            f.write(f"Final Training Loss: {result['final_train_loss']:.4f}\n")
            f.write(f"Final Validation Loss: {result['final_val_loss']:.4f}\n")
            f.write(f"Final BLEU Score: {result['final_bleu']:.2f}\n")
            f.write(f"Best Validation Loss: {result['best_val_loss']:.4f}\n")
            f.write(f"Average Epoch Time: {result['avg_epoch_time']:.2f}s\n\n")
        
        f.write("Overall Statistics\n")
        f.write("===================\n\n")
        f.write(f"Mean Training Loss: {mean(train_losses):.4f} ± {stdev(train_losses):.4f}\n")
        f.write(f"Mean Validation Loss: {mean(val_losses):.4f} ± {stdev(val_losses):.4f}\n")
        f.write(f"Mean BLEU Score: {mean(bleu_scores):.2f} ± {stdev(bleu_scores):.2f}\n")
        f.write(f"Mean Epoch Time: {mean(epoch_times):.2f}s ± {stdev(epoch_times):.2f}s\n\n")
        f.write(f"Best Model: {best_model['name']} with BLEU Score: {best_model['final_bleu']:.2f}\n")

if __name__ == "__main__":
    main()