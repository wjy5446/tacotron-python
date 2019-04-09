import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, query_size, context_size, hidden_size=None):
        super(Attention, self).__init__()

        if hidden_size is None:
            self.hidden_size = context_size

        self.W_q = nn.Linear(query_size, self.hidden_size, bias=False)
        self.W_c = nn.Linear(context_size, self.hidden_size, bias=False)
        self.v = nn.Parameter(torch.normal(mean=torch.zeros(self.hidden_size),
                                           std=torch.ones(self.hidden_size)))

    def forward(self, query, memory, memory_mask):
        """

        :param query: (batch_size, frame_length // r, input_size)
        :param memory: (batch_size, encoder_length, embed_size)
        :param memory_mask:
        :return:
        """
        batch_size = memory.size(0)
        seq_len = memory.size(1)
        memory_size = memory.size(2)

        frame_len = query.size(1)
        query_size = query.size(2)

        query_tiled = query.unsqueeze(2)
        query_tiled = query_tiled.repeat(1, 1, seq_len, 1)
        query_tiled = query_tiled.view(-1, query_size)

        memory_tiled = memory.unsqueeze(1)
        memory_tiled = memory_tiled.repeat(1, frame_len, 1, 1)
        memory_tiled = memory_tiled.view(-1, memory_size)

        info_matrix = torch.tanh(self.W_q(query) + self.W_c(memory))

        v_tiled = self.v.unsqueeze(0).repeat(batch_size * frame_len * seq_len, 1)

        energy = torch.sum(v_tiled * info_matrix, dim=1)
        energy = energy.view(batch_size, frame_len, seq_len) # (batch_size, frame_length, seq_length)

        energy = energy.float().masked_fill(memory_mask, float('-inf')).type_as(energy)
        alignment = F.softmax(energy.float(), dim=2).type_as(energy)

        return alignment

