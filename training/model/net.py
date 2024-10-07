import torch

# for reproducibility
torch.manual_seed(1)
import random

random.seed(1)


class CNN_LSTM(torch.nn.Module):
    """Two convolutional nets and a LSTM"""

    def __init__(
        self,
        kernel_size,
        n_filters1,
        n_filters2,
        pool_size,
        hidden_size,
        num_features,
        output_size,
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            num_features, n_filters1, kernel_size, padding="same"
        )
        self.conv2 = torch.nn.Conv1d(
            n_filters1, n_filters2, kernel_size, padding="same"
        )
        self.max_pool = torch.nn.MaxPool1d(pool_size)
        self.lstm = torch.nn.LSTM(n_filters2, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor):
        """Forward pass on the net

        Args:
            input (torch.Tensor): tensor with shape (N, C_in, L)

        Returns:
            tuple: hidden states and last hidden state
        """
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = torch.transpose(out, -2, -1)  # modify shape for lstm
        out, (hidden, _) = self.lstm(out)
        return out, self.output(hidden)
