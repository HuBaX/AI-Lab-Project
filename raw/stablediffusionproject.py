# -*- coding: utf-8 -*-
"""StableDiffusionProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fQn_cM3JRMzLeY-_kKh5_k3plHjq3FCM
"""

!pip install datasets

import numpy as np 
from PIL import Image
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

diffusiondb = load_dataset('poloclub/diffusiondb', 'large_first_1k')

train_df = pd.DataFrame(diffusiondb["train"])
train_df = train_df[["image", "prompt"]]
del diffusiondb
train_df

nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    return [tok.text for tok in nlp(text)]

tokenize("Hallo ich bin Lukas")

word_counts = Counter()
c = 0
for sentence in train_df["prompt"]:
    doc = nlp(sentence)
    
    # Iterate over each token in the processed sentence
    for token in doc:
        # Check if the token is a word (excluding punctuation and whitespace)
        if token.is_alpha:
            # Increment the count for the word
            word_counts[token.text] = word_counts.get(token.text, 0) + 1
    
    c += 1
    if c > 20: 
      break

# Print the word counts
for word, count in word_counts.items():
    print(f"{word}: {count}")

nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    return [tok.text for tok in nlp(text)]

class DiffusionDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.vocab_size, self.word2index = self.build_vocab()
        self.transformed_images = self.transform_images()
        self.tokenized_prompts = self.tokenize_and_index_prompts()
        self.eos_index = self.word2index["<EOS>"] 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.transformed_images[index]
        prompt = self.tokenized_prompts[index]

        # Pad the prompt to the length of the longest prompt in the batch
        max_length = max(len(p) for p in self.tokenized_prompts)
        padded_prompt = F.pad(torch.tensor(prompt), (0, max_length - len(prompt)), value=self.eos_index)
        #print(f"Padded Prompt Shape: {padded_prompt.shape}\nPadded Prompt: {padded_prompt}\n")
        return image, padded_prompt

    def build_vocab(self):
        word_counts = Counter()

        # Das zählt nur die Buchstaben und deren Häufigkeit
        #for tokens in self.data["prompt"]:
        #    word_counts.update(tokens)

        for sentence in train_df["prompt"]:
          doc = nlp(sentence)
          for token in doc:
            if token.is_alpha:
                word_counts[token.text] = word_counts.get(token.text, 0) + 1

        vocab = [word for word, count in word_counts.most_common(5000)]
        vocab_size = len(vocab) + 2  # Increment vocab_size by 2 for <UNK> and <EOS> tags

        word2index = {word: i+2 for i, word in enumerate(vocab)}  # Shift indices by 2 for <UNK> and <EOS>
        word2index["<UNK>"] = 0
        word2index["<EOS>"] = 1

        return vocab_size, word2index

    def transform_images(self):
      transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.ToTensor()])

      # Convert the PIL image to Torch tensor of size 512x512
      return self.data["image"].apply(transform).to_list()



    def tokenize_and_index_prompts(self):
        return self.data["prompt"].apply(self.tokenize).apply(self.tokens_to_indices).tolist()

    def tokenize(self, text):
      return [tok.text for tok in nlp(text)]

    def tokens_to_indices(self, tokens):
        return [self.word2index.get(word, 0) for word in tokens] + [self.word2index["<EOS>"]]

class ImagePromptGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, end_token_index):
        super(ImagePromptGenerator, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        for param in self.encoder.parameters():
          param.requires_grad = False
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_size)
        self.encoder.fc.requires_grad = True
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)  # Set batch_first=True to have input shape as (batch, seq_len, feature_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.end_token_index = end_token_index
        self.hidden_size = hidden_size
        self.num_layers = 1  # Adjust this value based on your requirements
        self.num_directions = 1  # Adjust this value based on your requirements

    def forward(self, images, prompts=None):
        features = self.encoder(images) # Shape [<batchSize>, 512]
        batch_size = images.size(0)
        hidden_state = self.init_hidden(batch_size)
        end_token_tensor = torch.tensor([[self.end_token_index]] * batch_size)  # Create end token tensor
        
        output_sequence = []
        sequence_finished = False
        device=features.device
        #print(device)

        for x in range(batch_size):
          output_sequence.append([])
        
        seq_count = 0
        while not sequence_finished:
            if seq_count > 0:
              lastPrediction = predicted_words.unsqueeze(1).unsqueeze(1)
              print(lastPrediction.shape)
              output, hidden_state = self.decoder(lastPrediction, hidden_state)  # Adjust the shape of features and hidden_state
            else:
              featureInput = features.unsqueeze(1)
              print(featureInput.shape)
              output, hidden_state = self.decoder(featureInput, hidden_state)  # Adjust the shape of features and hidden_state
            outputs = self.fc(output.squeeze(1))
            #output_sequence.append(outputs)
            
            _, predicted_words = outputs.max(dim=1)  # Get the predicted words
            #print(f"Predicted Words: {predicted_words}\n")
            for batch, output in enumerate(predicted_words):
              output_sequence[batch].append(output)

            seq_count += 1

            #if len(output_sequence) > 69:
            #  output_sequence.append(torch.tensor([self.end_token_index] * batch_size).unsqueeze(1))
            #  break

            if len(output_sequence[0]) > 69:
              for j in range(len(output_sequence)):
                output_sequence[j].append(torch.tensor(self.end_token_index).to(device))
              break            

            if (predicted_words == self.end_token_index).all():  # Check if all predicted words are the end token
              sequence_finished = True

        if prompts is not None:
          #print(f"Prompts Shape: {prompts.shape}")
          max_prompt_length = max(len(prompt) for prompt in prompts)
          #print(f"Max Prompt Length: {max_prompt_length}")
          output_sequence_padded = []
          #print(f"Output Sequence Shape: {len(output_sequence)}\n")
          for seq in output_sequence:
            #print(f"Seq Shape: {seq.shape}\nSeq: {seq}\n")
            #print(f"Seq Shape: {len(seq)}\nSeq: {seq}\n")
            #pad_length = max_prompt_length - seq.size(0)
            pad_length = max_prompt_length - len(seq)
            #print(f"Pad Length: {pad_length}")
            seq = torch.stack(seq).to(device)
            #print(f"Seq shape2: {seq.shape}")
            #pad_tensor = torch.full((pad_length,), self.end_token_index, device=seq.device)
            pad_tensor = torch.full((pad_length,), self.end_token_index, device=device)
            #print(f"Pad Tensor shape: {pad_tensor.shape}")
            #print(f"Pad Tensor Shape: {pad_tensor.shape}\nPad Tensor: {pad_tensor}")
            #padded_seq = torch.cat([seq, pad_tensor], dim=0)
            padded_seq = torch.cat([seq, pad_tensor], dim=0)
            #print(f"Pad seq shape: {padded_seq.shape}")
            output_sequence_padded.append(padded_seq)
      
          output_sequence_padded = torch.stack(output_sequence_padded, dim=1)
          return output_sequence_padded
        else:
          output_sequence = torch.stack(output_sequence, dim=1)
          return output_sequence



    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device))

embed_size = 512
hidden_size = 256

dataset = DiffusionDataset(train_df)
del train_df

dataset.vocab_size

c = 0
for img in dataset.transformed_images:
  print(img.size())
  c += 1
  if c > 5:
    break

model = ImagePromptGenerator(embed_size, hidden_size, dataset.vocab_size, dataset.eos_index)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {pytorch_total_params}")

batch_size = 2
num_epochs = 10

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

from tqdm import tqdm

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0

    # Wrap the data_loader with tqdm for progress bar
    with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
        for images, prompts in progress_bar:
            images, prompts = images.to(device), prompts.to(device)

            optimizer.zero_grad()

            outputs = model(images, prompts)

            #print(outputs)
            #print(outputs.shape)

            # Reshape prompts and outputs for computing loss
            #prompts_flat = prompts.view(-1)
            prompts_flat = prompts.float()
            #outputs_flat = outputs.view(-1, dataset.vocab_size)
            outputs_flat = outputs.transpose(0, 1).float()
            outputs_flat.requires_grad = True

            #print(prompts_flat)
            #print(outputs_flat)
            #print(prompts_flat.shape)
            #print(outputs_flat.shape)

            # Create a mask to ignore padded positions
            mask = (prompts_flat != dataset.word2index["<EOS>"]).float()

            # Compute loss only on non-padded positions
            loss = criterion(outputs_flat, prompts_flat) * mask

            # Calculate the average loss only over non-padded positions
            loss = loss.sum() / mask.sum()
            #print(loss.item())

            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Update progress bar description with the current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Print average loss for the epoch
    average_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")