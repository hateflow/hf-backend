import math
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class ToxicClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, output_size, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.ninp = embedding_dim
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, output_size)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        # self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.embedding(src)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.elu(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def data_process(dframe, equally=False):
    text, out = dframe['comment_text'], dframe.drop('comment_text', axis=1).values
    x_data, y_data = [], []
    # last = 1 - out[0]

    for comment, labels in zip(text, out):
        # if not equally or label != last:
        tokenized = tokenizer(comment)
        if not tokenized:
            tokenized = ["????????"]
        x_data.append(torch.tensor([vocab[token] for token in tokenized],
                                   dtype=torch.long))
        y_data.append(torch.tensor(labels, dtype=torch.long))
        # last = label
    y_data = torch.cat(y_data).reshape(-1, 6)

    return x_data, y_data


def train():
    global batches_trained
    model.train()
    start_time = time.time()

    for batch_start in range(0, len(y_train), batch_size):
        optimizer.zero_grad()

        loss = 0
        mean = 0
        n_comments = len(y_train)
        for inputs, target in zip(x_train, y_train):
            output = model(inputs.unsqueeze(0))
            # noinspection PyArgumentList,PyUnresolvedReferences
            label_pred = torch.max(output, axis=1).values[0].unsqueeze(1)
            loss += loss_fn(label_pred, target) / n_comments
            mean += torch.mean(label_pred).item() / n_comments
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        batches_trained += 1
        pierogi.plot_loss(epoch, batches_trained, loss.item(), "train")
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} | Batch {batch_start // batch_size + 1:2.0f}/{n_batches} | "
            f"lr {scheduler.get_last_lr()[0]:.3} | "
            f"{elapsed:2.2f}s | loss {loss.item():.3f} | mean {mean:2.3f}")
        scheduler.step()
        start_time = time.time()


def test(epoch=0):
    model.eval()
    correct = torch.zeros(y_test.shape)
    with torch.no_grad():
        for batch_start in range(0, len(y_test), batch_size):
            for chunk_idx in range(batch_start, min(batch_start + batch_size, len(y_test) - 1)):
                inputs, targets = x_test[chunk_idx], y_test[chunk_idx]
                outputs = model(inputs.unsqueeze(0))
                # noinspection PyArgumentList,PyUnresolvedReferences
                correct[chunk_idx] = ((targets > .5) == (torch.max(outputs, axis=1).values[0] > .5))

    val_loss = 1 - torch.mean(correct).item()  # (1 - (positive_loss + negative_loss) / 2)
    positive_loss = 1 - torch.mean(correct[y_test == 1.]).item()
    negative_loss = 1 - torch.mean(correct[y_test == 0.]).item()

    pierogi.plot_loss(epoch, batches_trained, positive_loss, "positive")
    pierogi.plot_loss(epoch, batches_trained, negative_loss, "negative")
    pierogi.plot_loss(epoch, batches_trained, val_loss, "test")
    if epoch:
        print('-' * 89)
        print(f"| Ende von Epoch {epoch} | {(time.time() - epoch_start_time):5.2f}s | val loss {val_loss:.4f} | "
              f"positive loss {positive_loss:.4f} | negative loss {negative_loss:.4f}")
        print('-' * 89)


if __name__ == '__main__':
    torch.set_num_threads(os.cpu_count() - 1)
    torch.set_deterministic(True)
    torch.manual_seed(42)

    print("Erzeuge Wörterbuch..")
    tokenizer = get_tokenizer('basic_english')

    train_dframe = pd.read_csv("../train_pretty.csv")[:5000]
    test_dframe = pd.read_csv("../test_pretty.csv")[:5000]
    vocab = build_vocab_from_iterator(map(tokenizer, train_dframe['comment_text']))

    print("Lade Daten..")
    # PARAM = 4
    x_train, y_train = data_process(train_dframe, equally=True)
    print("Training: ", len(y_train), "Kommentare")
    x_test, y_test = data_process(test_dframe, equally=True)
    print("Test: ", len(y_test), "Kommentare")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Erzeuge Modell..")
    MAX_BATCH_SIZE = 5000
    n_batches = math.ceil(len(x_train) / MAX_BATCH_SIZE)
    batch_size = math.ceil(len(x_train) / n_batches)

    vocab_size = len(vocab.stoi)
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    N_TRANSFORMER_LAYERS = 2
    N_HEADS = 2
    OUTPUT_SIZE = 6
    DROPOUT_P = 0.1
    model = ToxicClassifier(vocab_size, EMBEDDING_DIM, N_HEADS, HIDDEN_DIM, N_TRANSFORMER_LAYERS, OUTPUT_SIZE,
                            DROPOUT_P
                            ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    BASE_LR = 0.01
    MAX_LR = 0.08
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=BASE_LR,
        max_lr=MAX_LR,
        step_size_up=10 * n_batches,
    )

    N_EPOCHS = 200

    n_positives = torch.sum(y_test).item()
    n_negatives = len(y_test) - n_positives

    with Pierogi() as pierogi:
        batches_trained = 0
        test()

        for epoch in range(1, N_EPOCHS + 1):
            epoch_start_time = time.time()
            train()
            test(epoch)
            torch.save(model.state_dict(), f"model-{epoch}.pt")
