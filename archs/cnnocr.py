import torch.nn as nn

from archs.mambaout import MambaOut


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class CNNOCR(nn.Module):
    def __init__(
        self,
        num_classes=11,
        hidden_size=288,
        in_ch=3,
        depths=(3, 3, 15, 3),
        dims=(48, 96, 192, 288),
        drop_path_rate=0.025,
    ):
        super().__init__()
        self.feature = MambaOut(
            in_ch, depths=depths, dims=dims, drop_path_rate=drop_path_rate
        )
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(dims[-1], hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.feature(x)

        x = self.AdaptiveAvgPool(x.permute(0, 3, 1, 2))
        x = x.squeeze(3)
        x = self.SequenceModeling(x)

        x = self.fc(x)

        return x
