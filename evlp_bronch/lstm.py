import random

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        output, hidden = self.lstm(input)
        output = self.out(F.leaky_relu(output))
        return output, hidden


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_output: Tensor,
        encoder_hidden: tuple[Tensor, Tensor],
        target: Tensor,
        teacher_forcing_ratio: float,
    ) -> Tensor:
        sequence_length = target.size(0)
        decoder_input = target[:1]
        # decoder_input[0, 0] = 0
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(sequence_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)
            if teacher_forcing_ratio <= 0:
                teacher_forcing = False
            elif teacher_forcing_ratio >= 1:
                teacher_forcing = True
            else:
                teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = target[i : i + 1]
            if not teacher_forcing:
                decoder_input = torch.cat(
                    (decoder_output.detach(), decoder_input[:, 1:]), dim=1
                )
        return torch.cat(decoder_outputs, dim=0)

    def forward_step(
        self, input: Tensor, hidden: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        output, hidden = self.lstm(input, hidden)
        output = self.out(F.leaky_relu(output))
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        teacher_forcing_ratio: float = 0,
    ) -> Tensor:
        encoder_output, encoder_hidden = self.encoder(input)
        decoder_output = self.decoder(
            encoder_output, encoder_hidden, target, teacher_forcing_ratio
        )
        return decoder_output
