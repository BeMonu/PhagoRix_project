import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, cnn_output_dim=128, cnn_kernel_size=5):
        super(model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embed_dim, cnn_output_dim, cnn_kernel_size, padding=cnn_kernel_size // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_output_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        lstm_out, (hidden, _) = self.lstm(x)
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)

        return self.softmax(self.fc(final))