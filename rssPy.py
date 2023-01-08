import feedparser
import json
import string
import regex as re

feeds = [
    "https://partner-feeds.publishing.tamedia.ch/rss/24heures/la-une"
]

vocabulary = {}

def clean(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d', '', text)

    # Remove whitespace
    text = text.strip()

    text = ' '.join(text.split())
    return text

def text2words(text):
    return clean(text).split(' ')



for feed_url in feeds:
    feed = feedparser.parse(feed_url)
    news_list = feed.entries
    for news in news_list:
        words = text2words(news.title)
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1
    vocabulary = sorted(vocabulary.items(), key=lambda x: -x[1])

k = 30
print(len(vocabulary))

#print(json.dumps(entry, indent=4, ensure_ascii=False))



import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, dropout=0.1):
        super().__init__()

        # Create the embedding layer
        self.embedding = nn.Embedding(input_dim, d_model)

        # Create the encoder and decoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dropout),
            num_layers=1
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dropout),
            num_layers=1
        )

        # Create the final output layer
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # Embed the input sequences
        src = self.embedding(src)
        trg = self.embedding(trg)

        # Run the input sequences through the encoder and decoder layers
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, src_mask, trg_mask)

        # Apply the final output layer
        output = self.output_layer(decoder_output)

        return output


# Set the device to run on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model and move it to the device
model = Transformer(input_dim=10, output_dim=10, d_model=16, n_heads=4).to(device)

# Set the model to training mode
model.train()

# Define some input and target sequences
input_seq = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]]).to(device)
target_seq = torch.tensor([[1, 2, 3, 4, 6], [1, 2, 3, 6, 6]]).to(device)

# Run the model on the input and target sequences
output = model(input_seq, target_seq)

# Calculate the loss
loss = nn.CrossEntropyLoss()(output.view(-1, 10), target_seq.view(-1))
print(loss)