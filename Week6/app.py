import streamlit as st
import torch
import pickle
import os
import numpy as np
import random
import torch.nn.functional as F
import logging
import sys
import warnings

warnings.filterwarnings("ignore", message=".*packaging.*")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

from rnn_model import Seq2Seq, init_model, translate_sentence
from config import RNN_CONFIGS, TRANSFORMER_CONFIGS, SPECIAL_TOKENS, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
TRANSFORMER_AVAILABLE = True

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logger.info("Transformer modules loaded successfully")
    import transformer_model
    from transformer_model import calculate_bleu

except Exception as e:
    logger.warning(f"Warning when importing transformers: {str(e)}")

st.set_page_config(
    page_title="English-Vietnamese Translation",
    page_icon="🌐",
    layout="wide"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CACHE_DIR = os.path.join(os.getcwd(), "models")

def load_rnn_model(checkpoint_path, config):
    try:
        vocab_en_path = 'data/processed/vocab_en.pkl'
        vocab_vi_path = 'data/processed/vocab_vi.pkl'
        
        if not os.path.exists(vocab_en_path) or not os.path.exists(vocab_vi_path):
            raise FileNotFoundError(f"Vocabulary files not found at {vocab_en_path} or {vocab_vi_path}")
            
        with open(vocab_en_path, 'rb') as f:
            src_vocab = pickle.load(f)
        
        with open(vocab_vi_path, 'rb') as f:
            trg_vocab = pickle.load(f)
        
        logger.info(f"Loaded vocabularies: English: {len(src_vocab)} tokens, Vietnamese: {len(trg_vocab)} tokens")
        
        model_tuple = init_model(config, len(src_vocab), len(trg_vocab))
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"Successfully loaded RNN model from {checkpoint_path}")
        return model, src_vocab, trg_vocab
        
    except Exception as e:
        logger.error(f"Error loading RNN model: {str(e)}")
        raise e


def load_transformer_model(model_name):
    if not TRANSFORMER_AVAILABLE:
        logger.warning("Transformer modules not available, cannot load model")
        return None, None
    
    model_dir = f"models/{model_name}"
    checkpoint_path = f"{model_dir}/best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}, falling back to pretrained model")
        try:
            logger.info(f"Loading pretrained transformer model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=CACHE_DIR, use_fast=False
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=CACHE_DIR
            )
            model.to(device)
            model.eval()
            logger.info(f"Successfully loaded pretrained transformer model: {model_name}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Error loading pretrained transformer model: {str(e)}")
            return None, None
    
    try:
        logger.info(f"Loading fine-tuned transformer model from checkpoint: {checkpoint_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=CACHE_DIR, use_fast=False
        )
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        logger.info(f"Successfully loaded fine-tuned transformer model from checkpoint (epoch {checkpoint['epoch']})")
        if 'val_loss' in checkpoint and 'bleu' in checkpoint:
            logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}, BLEU score: {checkpoint['bleu']:.4f}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading fine-tuned transformer model: {str(e)}")
        return None, None

def translate_with_rnn(model, src_vocab, trg_vocab, text, max_length=50):
    try:
        logger.info(f"Translating with RNN model: '{text[:30]}...'")
        trg_tokens = translate_sentence(
            model=model,
            sentence=text,
            src_vocab=src_vocab,
            trg_vocab=trg_vocab,
            device=device,
            max_len=max_length
        )
        
        translated_text = ' '.join(trg_tokens)
        logger.info(f"Translation completed: '{translated_text[:30]}...'")
        return translated_text
    except Exception as e:
        logger.error(f"Error during RNN translation: {str(e)}")
        return f"Translation error: {str(e)}"


def translate_with_transformer(model, tokenizer, text, max_length=50):
    if model is None or tokenizer is None:
        return "Transformer model not available"
        
    try:
        logger.info(f"Translating with Transformer model: '{text[:30]}...'")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                early_stopping=True
            )
        
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Translation completed: '{translated_text[:30]}...'")
        return translated_text
    except Exception as e:
        logger.error(f"Error during Transformer translation: {str(e)}")
        return f"Translation error: {str(e)}"


def main():
    st.title("🌐 English to Vietnamese Translation")
    st.markdown("""This app translates English text to Vietnamese using neural machine translation models.""")
    
    st.sidebar.title("Model Selection")
    
    model_options = ["RNN (Seq2Seq with Attention)"]
    if TRANSFORMER_AVAILABLE:
        model_options.append("Transformer (Fine-tuned)")
    
    model_type = st.sidebar.radio(
        "Select Model Type",
        model_options
    )
    
    if model_type == "RNN (Seq2Seq with Attention)":
        st.sidebar.markdown("""**RNN Model**: Sequence-to-sequence model with attention mechanism using GRU cells.""")
    elif model_type == "Transformer (Fine-tuned)":
        st.sidebar.markdown("""**Transformer Model**: Pre-trained transformer model fine-tuned on English-Vietnamese data.""")
    
    if model_type == "RNN (Seq2Seq with Attention)":
        rnn_checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
        
        if not rnn_checkpoints:
            st.error("No RNN checkpoints found in the 'checkpoints' directory.")
            return
        
        selected_checkpoint = st.sidebar.selectbox(
            "Select RNN Checkpoint",
            rnn_checkpoints
        )
        
        config_name = selected_checkpoint.split("_final.pt")[0]
        selected_config = None
        
        for config in RNN_CONFIGS:
            if config["name"] == config_name:
                selected_config = config
                break
        
        if selected_config is None:
            st.error(f"Configuration for {config_name} not found.")
            return
        
        st.sidebar.subheader("Model Configuration")
        st.sidebar.json(selected_config)
        
        with st.spinner("Loading RNN model..."):
            model, src_vocab, trg_vocab = load_rnn_model(
                f"checkpoints/{selected_checkpoint}", 
                selected_config
            )
        
        st.success("RNN model loaded successfully!")
        
        st.subheader("Enter English Text")
        input_text = st.text_area("English text to translate", height=150)
        
        if st.button("Translate"):
            if input_text:
                with st.spinner("Translating..."):
                    translated_text = translate_with_rnn(model, src_vocab, trg_vocab, input_text)
                
                st.subheader("Vietnamese Translation")
                st.write(translated_text)
            else:
                st.warning("Please enter some text to translate.")
    
    elif model_type == "Transformer (Fine-tuned)" and TRANSFORMER_AVAILABLE:
        available_models = []
        for dir_name in os.listdir("models"):
            if dir_name.startswith("Transformer_Fine-tune_") and os.path.isdir(os.path.join("models", dir_name)):
                if os.path.exists(os.path.join("models", dir_name, "best_model.pt")):
                    available_models.append(dir_name)
        
        config_models = list(set([config["model_name"] for config in TRANSFORMER_CONFIGS]))
        
        all_models = available_models + [model for model in config_models if model not in available_models]
        
        if not all_models:
            st.error("No Transformer models found in models directory or configuration.")
            return
            
        selected_model = st.sidebar.selectbox(
            "Select Transformer Model",
            all_models,
            index=0 if available_models else len(available_models)
        )
        
        st.sidebar.subheader("Model Information")
        
        is_fine_tuned = selected_model in available_models
        
        if is_fine_tuned:
            st.sidebar.markdown(f"**Model**: {selected_model} (Fine-tuned)")
            
            try:
                checkpoint_path = os.path.join("models", selected_model, "best_model.pt")
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    st.sidebar.markdown(f"**Epoch**: {checkpoint.get('epoch', 'N/A')}")
                    st.sidebar.markdown(f"**Validation Loss**: {checkpoint.get('val_loss', 'N/A'):.4f}")
                    st.sidebar.markdown(f"**BLEU Score**: {checkpoint.get('bleu', 'N/A'):.4f}")
            except Exception as e:
                logger.warning(f"Could not load metrics from checkpoint: {str(e)}")
        else:
            st.sidebar.markdown(f"**Model**: {selected_model} (Pretrained)")
            
            matching_configs = [config for config in TRANSFORMER_CONFIGS 
                               if config["model_name"] == selected_model]
            
            if matching_configs:
                selected_config = matching_configs[0]
                if "description" in selected_config:
                    st.sidebar.markdown(f"**Description**: {selected_config['description']}")
        
        try:
            with st.spinner("Loading Transformer model..."):
                model, tokenizer = load_transformer_model(selected_model)
                
                if model is None:
                    st.error("Failed to load Transformer model.")
                    st.info("Please try selecting the RNN model instead.")
                    return
            
            if selected_model in available_models:
                st.success(f"Fine-tuned Transformer model loaded successfully!")
            else:
                st.success(f"Pretrained Transformer model loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading Transformer model: {str(e)}")
            st.info("Please try selecting the RNN model instead.")
            return
        
        st.subheader("Enter English Text")
        input_text = st.text_area("English text to translate", height=150)
        
        if st.button("Translate"):
            if input_text:
                with st.spinner("Translating..."):
                    translated_text = translate_with_transformer(model, tokenizer, input_text)
                
                st.subheader("Vietnamese Translation")
                st.write(translated_text)
            else:
                st.warning("Please enter some text to translate.")
    else:
        st.error("Transformer model is not available. Please install the required dependencies.")
        st.info("Try: pip install transformers packaging")

if __name__ == "__main__":
    main()