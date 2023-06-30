# %%
#!pip install datasets

# %%
import numpy as np 
from datasets import load_dataset
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import spacy
from random import seed
from random import random
import torchtext
import pickle
import os
import matplotlib.pyplot as plt

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
data_path = "preprocessed"
word2index_path = "word2index.pkl"

# %%
class DiffusionDataset(Dataset):
    def __init__(self, data_path, word2index_path, train_split_ratio=0.8, train=True):
        self.data_path = data_path
        with open(word2index_path, 'rb') as f:
            self.word2index = pickle.load(f)
        self.vocab_size = len(self.word2index)
        self.eos_index = self.word2index["<EOS>"]
        self.train = train
        self.train_split_ratio = train_split_ratio
        self.files = os.listdir(data_path)

        n_image_prompt = int(len(self.files) / 2) # len(self.files) must be an even number

        # Calculate the split index
        self.split_index = int((self.train_split_ratio * n_image_prompt))
        
        # Calculate the total number of rows
        self.total_rows = 0
        self.raw_prompts = pd.read_csv("raw_prompts.csv")["prompt"]

        if self.train:
            self.total_rows = self.split_index
        else:
            self.total_rows = n_image_prompt - self.split_index

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if not self.train:
                idx += self.split_index - 1
            image = np.load(f"{data_path}\\image_{idx}.npy")
            prompt = np.load(f"{data_path}\\prompt_{idx}.npy")
            return torch.tensor(image), torch.tensor(prompt), self.raw_prompts[idx]
        else:
            image_batch = []
            prompt_batch = []
            if not self.train:
                idx = np.array(idx)
                idx += self.split_index - 1
            for i in idx:
                image = np.load(f"{data_path}\\image_{i}.npy")
                prompt = np.load(f"{data_path}\\prompt_{i}.npy")
                image_batch.append(image)
                prompt_batch.append(prompt)
                return torch.tensor(image_batch), torch.tensor(prompt_batch), self.raw_prompts[idx]

# %%
class EncoderCNN(nn.Module):
    def __init__(self, n_layers, hid_dim):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hid_dim)

        self.hidden_size = hid_dim
        self.n_layers = n_layers

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)

        batch_size = features.size(0)
        hidden = features.unsqueeze(0).expand(self.n_layers, batch_size, self.hidden_size)
        # Initialize the cell state with zeros
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(features.device)
        return hidden, cell



class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, n_layers, hid_dim, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden.contiguous(), cell.contiguous()))
        
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell

# %%
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
       
    def forward(self, src, trg, teacher_forcing_ratio = float(0.5)):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[:, 0]

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token)
            if self.train:
                input = trg[:,t] if random() < teacher_forcing_ratio else top1
            else:
                input = top1
        return outputs

# %%
dataset = DiffusionDataset(data_path, word2index_path, train_split_ratio=0.8, train=True)

# %%
embed_size = 512
hidden_size = 256
output_size = dataset.vocab_size
n_layers = 2
dec_dropout = 0.5

batch_size = 64
num_epochs = 4
clip = 1

# seed random number generator
seed(1)

# %%
dataset.vocab_size

# %%
encoder = EncoderCNN(n_layers, hidden_size).to(device)
decoder = DecoderRNN(output_size, embed_size, n_layers, hidden_size, dec_dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# %%
def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, prompts, raw_prompts = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge prompts (from tuple of 1D tensor to 2D tensor).
    lengths = [len(prompt) for prompt in prompts]
    padded_prompts = torch.full((len(prompts), max(lengths)), dataset.word2index["<PAD>"])
    #padded_prompts = torch.zeros(len(prompts), max(lengths)).long()
    for i, cap in enumerate(prompts):
        end = lengths[i]
        padded_prompts[i, :end] = cap[:end]

    return images, padded_prompts, raw_prompts, lengths

# %%
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# %%
x = [ x[1] for x in next(iter(data_loader)) ]
x[1]

# %%
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# %%
#TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
#TODO
#criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# %%
def translate_output(output, word2index):
    index2word = {index: word for word, index in word2index.items()}  # Create index-to-word dictionary
    translated_sentences = []
    for seq in output:
        sentence = []
        for idx in seq:
            word = index2word.get(idx.item(), "<UNK>")
            if word == "<EOS>":
                break
            sentence.append(word)
        translated_sentence = " ".join(sentence)
        translated_sentences.append(translated_sentence)
    return translated_sentences

# %%
# for i, (images, prompts, trg_lengths) in enumerate(data_loader):
#     example_images = images
#     example_prompts = prompts
#     break

# # Translate the example prompt
# translated_example_prompts = translate_output(prompts, dataset.word2index)


# %%
def get_translations(images, prompts): 
    model.eval()
    with torch.no_grad():
        # Move images and prompts to the device
        images = images.to(device)
        prompts = prompts.to(device)

        # Perform forward pass for the images and prompts
        outputs = model(images, prompts)

        # Get the predicted words with the highest probability
        top1 = outputs.argmax(2).transpose(0, 1)

        # Translate the predicted output to words
        translated_output = translate_output(top1, dataset.word2index)

        return translated_output
    

# %%
import sys
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)

# %%
# Training loop
translations_list = []  # List to store translated sentences

losses = []

for epoch in range(num_epochs):
    model.train()
    for i, (images, sampled_prompts, raw_prompts, trg_lengths) in enumerate(data_loader):
        images = images.to(device)
        sampled_prompts = sampled_prompts.to(device)

        # TODO add packing?
        #targets = pack_padded_sequence(prompts, trg_lengths, batch_first=True)[0]

        optimizer.zero_grad()
        output = model(images, sampled_prompts)

        # Remove the <sos> token and reshape the output and target tensors
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim).contiguous()

        trg = sampled_prompts.transpose(0, 1)[1:].contiguous().view(-1)
        #output_indices = np.argmax(output.cpu().detach().numpy(), axis=1)

        #print(f"Output Prompt: {output_indices}\nOutput Prompt Length: {len(output_indices)},\nTarget Prompt: {trg}\nTarget Prompt Length: {len(trg)}")

        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses.append(loss.item())
        if i % batch_size == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch + 1, num_epochs, i, len(data_loader), loss.item(), np.exp(loss.item())))

    # Get translations after each epoch and append to the list
    #translations_list.append(get_translations(example_images, example_prompts))

# %%
# Plot the losses
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# %%
averaged_losses = []
for i in range(batch_size, len(losses), batch_size):
    averaged_losses.append(np.array(losses[i-batch_size:i]).mean())


# Plot the losses
plt.plot(np.arange(0, len(losses)-batch_size, batch_size), averaged_losses)
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# %%
len(losses)

# %%
torch.save(model.state_dict(), 'parameters/seq2seq_removed_punctuation_5_lstm_layers.pth')

# %%
encoder = EncoderCNN(n_layers, hidden_size).to(device)
decoder = DecoderRNN(output_size, embed_size, n_layers, hidden_size, dec_dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load("parameters/seq2seq_removed_punctuation.pth"))
model.eval()

# %%
test_dataset = DiffusionDataset(data_path, word2index_path, train_split_ratio=0.8, train=False)
test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fn)

# %%
index2word = {index: word for word, index in test_dataset.word2index.items()}

# %%
def translate_plain(output, index2word):
    translated_sentences = []
    for seq in output:
        sentence = []
        for idx in seq:
            word = index2word.get(idx.item(), "<UNK>")
            if word != "<EOS>":
                sentence.append(word)
        
        translated_sentences.append(" ".join(sentence))
    return translated_sentences

# %%
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to embed a word using spaCy
def embed_word(word):
    return nlp(str(word)).vector

# Function to embed a batch of sentences with variable sizes
def embed_batch(batch):
    embedded_batch = []
    for sentence in batch:
        embedded_sentence = []
        for word in sentence:
            embedded_word = embed_word(word)
            embedded_sentence.append(embedded_word)
        embedded_batch.append(embedded_sentence)
    return embedded_batch

# %%
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from random import random
st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
model.eval()  # Set the model to evaluation mode
total_cosine_similarity = 0
total_examples = 0

sampled_images = []
sampled_translated_outputs = []
sampled_raw_prompts = []
with torch.no_grad():
    for i, (images, sampled_prompts, raw_prompts, trg_lengths) in enumerate(test_loader):
        images = images.to(device)
        sampled_prompts = sampled_prompts.to(device)

        output = model(images, sampled_prompts)
        
        output_sentences = output.argmax(2).transpose(0, 1)
        translated_output = translate_plain(output_sentences, index2word)


        #print(f"Translated Output: {translated_output[0]},\n Actual prompt: {raw_prompts[0]}")
        embedded_output = st_model.encode(translated_output)


        embedded_prompts = st_model.encode(raw_prompts)
        #print(f"Translated Output: {translated_output[0]},\n Actual Prompt: {raw_prompts[0]}")
        cosine_similarities = cosine_similarity(embedded_output, embedded_prompts)
        mean_cosine_similarity = cosine_similarities.mean()

        total_cosine_similarity += mean_cosine_similarity
        total_examples += 1

        if random() > 0.8:
            sampled_images.append(images.cpu()[0])
            sampled_translated_outputs.append(translated_output[0])
            sampled_raw_prompts.append(raw_prompts[0])

        if i % 1 == 0:
            print('Test Step [{}/{}], Mean Cosine Similarity: {:5.4f}'
                  .format(i, len(test_loader), mean_cosine_similarity))

average_cosine_similarity = total_cosine_similarity / total_examples
print('Average Mean Cosine Similarity: {:5.4f}'.format(average_cosine_similarity))

# %%
from matplotlib import pyplot as plt

num_images = len(sampled_images)  # Update this value to the total number of images
fig, axs = plt.subplots(num_images, 1, figsize=(8, 8 * num_images))  # num_images rows, 1 column

for i, image in enumerate(sampled_images):
    image = np.transpose(image, (1, 2, 0))  # Transpose dimensions from (3, 512, 512) to (512, 512, 3)
    raw_prompt = sampled_raw_prompts[i]
    output_prompt = sampled_translated_outputs[i]

    embedded_output = st_model.encode(output_prompt)


    embedded_prompts = st_model.encode(raw_prompt)

    similarity = cosine_similarity(embedded_output.reshape(1, -1), embedded_prompts.reshape(1, -1))

    ax = axs[i]  # Select the current subplot
    ax.set_title(f"Prompt:\n{raw_prompt}\n\n Output of Model:\n{output_prompt}\nCosine Similarity:{similarity} ")
    ax.imshow(image, interpolation="nearest")

plt.tight_layout()
plt.show()


