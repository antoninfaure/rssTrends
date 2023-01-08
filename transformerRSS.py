import feedparser
import regex as re
import gensim
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torchtext
import json
import numpy as np

feeds = [
    #"https://partner-feeds.publishing.tamedia.ch/rss/24heures/la-une",
    "http://www.lemonde.fr/rss/une.xml",
    "https://www.france24.com/fr/actualites/rss",
    "http://www.lefigaro.fr/rss/figaro_actualites.xml",
    "https://www.franceinter.fr/rss",
    "http://www.lexpress.fr/rss/une.xml",
    "http://www.liberation.fr/rss/",
    "https://www.la-croix.com/RSS",
    "https://www.nouvelobs.com/syndication/all.rss"
]

# Set the device to run on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, embed_dim):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout =dropout),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout =dropout),
            num_layers
        )

    def forward(self, input_sequences, output_sentences):
        # Apply the encoder to the sequences
        encoder_output = self.encoder(input_sequences)

        # Apply the decoder to the target sequences
        decoder_output = self.decoder(output_sequences, encoder_output)
        return decoder_output

def parsetext(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\d', '', text)
    text = text.strip()
    text = ' '.join(text.split())

    return text.split()
    

sentences = []
for feed_url in feeds:
    feed = feedparser.parse(feed_url)
    news_list = feed.entries
    for news in news_list:
        sentences.append(news.title)
        if 'summary' in news.keys():
            sentences.append(news.summary)

print(len(sentences))

sequences = [parsetext(sentence) for sentence in sentences]  
words = [word for seq in sequences for word in seq]  # Flatten the list of sequences into a list of words
vocab = list(set(words))

def counts(sequences):
    counts = {}
    for seq in sequences:
        for word in seq:
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1
    
    return counts
print(len(vocab))

#print(sorted(counts(sequences).items(), key=lambda x: -x[1]))

def find(word, sequences):
    results = []
    for seq in sequences:
        if word in seq:
            results.append(" ".join(seq))
    return results

for seq in find('ukraine', sequences):
    print(seq)
""" 
embed_dim = 100  # size of the word embedding

# Train the Word2Vec model
word2vec = Word2Vec(sequences, vector_size=embed_dim, window=5, min_count=1, workers=4)

# Create a list of word indices
word_indices = [[vocab.index(word) for word in seq] for seq in sequences]

# Create a list of word embeddings
word_embeddings = []
for word in vocab:
    # Replace this with your word embedding code
    word_embeddings.append(word2vec.wv[word])

# Convert word_embeddings to a list of PyTorch tensors
word_embeddings_tensor = [torch.from_numpy(array) for array in word_embeddings]

# Convert the sequences to a tensor of shape (batch_size, sequence_length, input_size)
batch_size = len(sequences)
input_size = embed_dim
sequence_length = max(len(seq) for seq in sequences)

sequences_tensor = torch.zeros(batch_size, sequence_length, input_size)
for i, seq in enumerate(word_indices):
    for j, word_index in enumerate(seq):
        sequences_tensor[i, j, :] = word_embeddings_tensor[word_index]

# Define the hyperparameters
hidden_size = 128
num_layers = 4
num_heads = 5
dropout = 0.1

# Make sure the embedding size is divisible by the number of heads
assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

# Create a transformer model
model = TransformerModel(input_size, hidden_size, num_layers, num_heads, dropout, embed_dim)

# Define the loss function and the optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

input_sequences = sequences_tensor
output_sequences = sequences_tensor


def postprocess_output(output, vocabulary, word_embedding):
    # Get the indices of the highest-scoring words
    word_scores, word_indices = output.max(dim=2)

    # Convert the indices to words
    words = []
    for i in range(word_indices.size(0)):
        for j in range(word_indices.size(1)):
            index = word_indices[i, j]
            word = vocabulary[index]
            words.append(word)

    # Use the word embedding to find similar words for any unknown words
    for i, word in enumerate(words):
        if word == "<unk>":
            # Get the vector for the unknown word
            vector = output[i, :]

            # Find the top-scoring words that are not unknown
            similar_words = word_embedding.similar_by_vector(vector, topn=1, restrict_vocab=None)
            words[i] = similar_words[0][0]

    return words

# Train the model
for epoch in range(100):
    # Reset the gradients
    optimizer.zero_grad()

    # Generate the output sequences
    output = model(input_sequences, output_sequences)

    # Compute the loss
    loss = loss_fn(output, output_sequences)

    # Backpropagate the error
    loss.backward()

    # Update the parameters
    optimizer.step()

    print(loss) """