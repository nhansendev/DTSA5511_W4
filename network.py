import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from constants import RNG


def get_num_params(model):
    # Get the number of parameters in the neural network
    SD = model.state_dict()
    num_params = 0
    for v in SD.values():
        num_params += np.prod(v.shape)
    return num_params


class LSTM_Head(nn.Module):
    def __init__(
        self, in_size=300, hidden_size=100, out_size=50, rnn_layers=5, dropout=0.3
    ) -> None:
        super().__init__()

        self.L1 = nn.LSTM(in_size, hidden_size, rnn_layers, batch_first=True)
        self.L2 = nn.Linear(hidden_size, out_size)
        self.act = nn.Hardswish()
        self.drop = nn.Dropout(dropout)

    def forward(self, X, indexes):
        A, _ = self.L1(X)

        return self.drop(self.act(self.L2(self.act(A[np.arange(X.shape[0]), indexes]))))


class GRU_Head(nn.Module):
    def __init__(
        self, in_size=300, hidden_size=100, out_size=50, rnn_layers=5, dropout=0.3
    ) -> None:
        super().__init__()

        self.L1 = nn.GRU(in_size, hidden_size, rnn_layers, batch_first=True)
        self.L2 = nn.Linear(hidden_size, out_size)
        self.act = nn.Hardswish()
        self.drop = nn.Dropout(dropout)

    def forward(self, X, indexes):
        A, _ = self.L1(X)

        return self.drop(self.act(self.L2(self.act(A[np.arange(X.shape[0]), indexes]))))


class BasicRNNet(nn.Module):
    def __init__(
        self,
        in_size=300,
        hidden_size=100,
        mid_size=50,
        out_size=1,
        rnn_layers=5,
        dropout=0.3,
        use_LSTM=True,
    ) -> None:
        super().__init__()

        if use_LSTM:
            self.RNN_head = LSTM_Head(
                in_size, hidden_size, mid_size, rnn_layers, dropout
            )
        else:
            self.RNN_head = GRU_Head(
                in_size, hidden_size, mid_size, rnn_layers, dropout
            )

        self.Dense_head = nn.Sequential(
            nn.Linear(in_size * 2, mid_size),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )

        self.out_net = nn.Sequential(
            nn.Linear(mid_size * 2, 10), nn.Hardswish(), nn.Linear(10, out_size)
        )

    def forward(self, X, KL, indexes):
        # Process sentences
        A = self.RNN_head(X, indexes)
        # Process keywords and labels
        B = self.Dense_head(KL)
        # Combine
        return self.out_net(torch.cat([A, B], 1))


def MAV(values, L=10):
    # Moving average
    out = [values[0]]
    for v in values[1:]:
        out.append(out[-1] + (v - out[-1]) / L)
    return out


def make_batch(indexes, batch_size=128, resampling_weights=None):
    # Generate a batch using indexes, with optional resampling weights
    return np.array(
        [
            indexes[k]
            for k in RNG.choice(len(indexes), batch_size, False, p=resampling_weights)
        ]
    )


def train_model(
    model,
    encoded_sentences,
    encoded_KL,
    length_indexes,
    train_targets,
    valid_targets,
    train_idxs,
    valid_idxs,
    resampling_weights=None,
    epochs=9000,
    batch_size=128,
    lr=0.002,
):
    opt = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.BCEWithLogitsLoss()

    loss_hist = []
    valid_hist = []
    best_params = None
    best_valid_score = None
    for i in tqdm(range(epochs)):
        # One batch
        opt.zero_grad()
        idxs = make_batch(train_idxs, batch_size, resampling_weights)

        res = model(encoded_sentences[idxs], encoded_KL[idxs], length_indexes[idxs])
        loss = loss_fn(res, train_targets[idxs])

        loss_hist.append(loss.item())
        loss.backward()
        opt.step()

        if i % 50 == 0:
            # Validation
            with torch.no_grad():
                model.eval()
                preds = (
                    model(
                        encoded_sentences[valid_idxs],
                        encoded_KL[valid_idxs],
                        length_indexes[valid_idxs],
                    )
                    .cpu()
                    .numpy()
                    >= 0.5
                )
                score = f1_score(valid_targets[valid_idxs], preds)
                if best_valid_score is None or score > best_valid_score:
                    best_valid_score = score
                    best_params = model.state_dict()
                valid_hist.append([i, score])
                model.train()

    # Use the best set of model parameters per validation scores
    model.load_state_dict(best_params)

    return model, loss_hist, valid_hist


def plot_model_performance(
    loss_hist,
    valid_hist,
    model,
    encoded_sentences,
    encoded_KL,
    length_indexes,
    valid_targets,
    valid_idxs,
    title="",
):
    fig, ax = plt.subplots(1, 2)
    axs = fig.axes
    fig.set_size_inches(12, 4)

    if len(title) > 0:
        plt.suptitle(title)

    # Plot loss and validation score history
    X, scores = list(zip(*valid_hist))
    axs[0].plot(loss_hist)
    axs[0].plot(X, scores, ".-")
    axs[0].plot(MAV(loss_hist, 50), "k")
    axs[0].legend(["BCELoss", "F1-Score"])
    axs[0].grid()

    # Generate confusion matrix
    with torch.no_grad():
        model.eval()
        preds = (
            model(
                encoded_sentences[valid_idxs],
                encoded_KL[valid_idxs],
                length_indexes[valid_idxs],
            )
            .cpu()
            .numpy()
            >= 0.5
        )
        targets = valid_targets[valid_idxs]
        model.train()

    score = np.max(scores)

    print(f"Validation F1 Score: {score:.3f}")

    cm = confusion_matrix(targets, preds)
    cplt = ConfusionMatrixDisplay(cm, display_labels=["No Disaster", "Disaster"])
    cplt.plot(ax=axs[1])
    plt.show()

    return score
