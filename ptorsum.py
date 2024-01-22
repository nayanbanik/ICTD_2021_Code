#necessary library imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import TabularDataset
from torchtext.data import Field, BucketIterator
import numpy as np
import random
import math
import time
import os
#import pandas as pd
import pickle
import re
import string

def cleaning_text(org_text):
    with open("./static/assets/spchar.txt", 'r', encoding='utf-8') as f:
        sp_char = f.read()
    sp_char = sp_char.split('।')
    stand_punct = [ch for ch in string.punctuation]
    sp_char = sp_char+stand_punct
    cleaned_text = ''
    for ch in org_text:
        if ch in sp_char:
            cleaned_text = cleaned_text+' '
        else:
            cleaned_text = cleaned_text+ch
    #print(cleaned_text)
    cleaned_text = re.sub('[a-zA-Z0-9]+', ' ', cleaned_text)
    cleaned_text = re.sub('।', ' । ', cleaned_text)
    cleaned_text = re.sub('‘', '', cleaned_text)
    cleaned_text = re.sub('’', '', cleaned_text)
    cleaned_text = re.sub('–', ' ', cleaned_text)
    cleaned_text = re.sub('\s+', ' ', cleaned_text)
    #print(cleaned_text)
    return cleaned_text

def eval_one(bd):
    #assign gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('./static/assets/vocab.pkl', 'rb') as f:
        vc = pickle.load(f)
    
    SRC = vc['src']
    TRG = vc['trg']
    #initialize global configs

    #initialize global configs
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    N_LAYERS = 3
    ENC_DROPOUT = 0.10
    DEC_DROPOUT = 0.10

    class Encoder(nn.Module):
        def __init__(self, input_dim, emb_dim, hid_dim, dropout):
            super().__init__()
            self.hid_dim = hid_dim
            self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!
            self.rnn = nn.GRU(emb_dim, hid_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, src):
            #src = [src len, batch size]
            embedded = self.dropout(self.embedding(src))
            #embedded = [src len, batch size, emb dim]
            outputs, hidden = self.rnn(embedded) #no cell state!
            #outputs = [src len, batch size, hid dim * n directions]
            #hidden = [n layers * n directions, batch size, hid dim]
            #outputs are always from the top hidden layer
            return hidden

    class Decoder(nn.Module):
        def __init__(self, output_dim, emb_dim, hid_dim, dropout):
            super().__init__()
            self.hid_dim = hid_dim
            self.output_dim = output_dim
            self.embedding = nn.Embedding(output_dim, emb_dim)
            self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
            self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, input, hidden, context):
            #input = [batch size]
            #hidden = [n layers * n directions, batch size, hid dim]
            #context = [n layers * n directions, batch size, hid dim]
            #n layers and n directions in the decoder will both always be 1, therefore:
            #hidden = [1, batch size, hid dim]
            #context = [1, batch size, hid dim]
            input = input.unsqueeze(0)
            #input = [1, batch size]
            embedded = self.dropout(self.embedding(input))
            #embedded = [1, batch size, emb dim]
            emb_con = torch.cat((embedded, context), dim = 2)
            #emb_con = [1, batch size, emb dim + hid dim]
            output, hidden = self.rnn(emb_con, hidden)
            #output = [seq len, batch size, hid dim * n directions]
            #hidden = [n layers * n directions, batch size, hid dim]
            #seq len, n layers and n directions will always be 1 in the decoder, therefore:
            #output = [1, batch size, hid dim]
            #hidden = [1, batch size, hid dim]
            output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim = 1)
            #output = [batch size, emb dim + hid dim * 2]
            prediction = self.fc_out(output)
            #prediction = [batch size, output dim]
            return prediction, hidden

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
            assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"

        def forward(self, src, trg, teacher_forcing_ratio = 0.5):
            #src = [src len, batch size]
            #trg = [trg len, batch size]
            #teacher_forcing_ratio is probability to use teacher forcing
            #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
            batch_size = trg.shape[1]
            trg_len = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
            #tensor to store decoder outputs
            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
            #last hidden state of the encoder is the context
            context = self.encoder(src)
            #context also used as the initial hidden state of the decoder
            hidden = context
            #first input to the decoder is the <sos> tokens
            input = trg[0,:]
            for t in range(1, trg_len):
                #insert input token embedding, previous hidden state and the context state
                #receive output tensor (predictions) and new hidden state
                output, hidden = self.decoder(input, hidden, context)
                #place predictions in a tensor holding predictions for each token
                outputs[t] = output
                #decide if we are going to use teacher forcing or not
                teacher_force = random.random() < teacher_forcing_ratio
                #get the highest predicted token from our predictions
                top1 = output.argmax(1) 
                #if teacher forcing, use actual next token as next input
                #if not, use predicted token
                input = trg[t] if teacher_force else top1
            return outputs
        
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    
    model.load_state_dict(torch.load('./static/assets/model.pt', map_location=torch.device('cpu')))
    model.eval()
    
    
    s = torch.tensor([SRC.vocab.stoi[i] for i in bd.split()]).to(device)
    s = s.view(-1, 1)
    #print(s.shape)
    t = torch.zeros((9, 1), dtype=torch.int64).to(device)
    t[0,:]=2
    #print(t.shape)

    output = model(s, t, 0) #turn off teacher forcing
    #print('output : ', output.shape)
    #trg = [trg len, batch size]
    #output = [trg len, batch size, output dim]

    toks = []
    for i in range(9):
        toks.append(output[i].argmax(1).squeeze().item())
    hd = ' '.join([TRG.vocab.itos[x] for x in toks[1:] if x not in [0,3]])
    if hd=='\u200d':
        return "Try Again..."
    else:
        return hd
     
    
if __name__ == "__main__":
    print('import as main')

