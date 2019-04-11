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
        :param query: mel data (batch_size, audio_len // r, input_dim)
        :param memory: encoder output (batch_size, text_len, embed_dim)
        :param memory_mask: mask (batch_size, audio_len, text_len)
        :return: align (batch_size, audio_len, text_len)
        """


        batch_size = memory.size(0)
        text_len = memory.size(1)
        embed_dim = memory.size(2)

        audio_len = query.size(1)
        input_dim = query.size(2)

        query_tiled = query.unsqueeze(2)
        query_tiled = query_tiled.repeat(1, 1, text_len, 1)
        query_tiled = query_tiled.view(-1, input_dim) # (batch_size * audio_len * text_len, input_dim)

        memory_tiled = memory.unsqueeze(1)
        memory_tiled = memory_tiled.repeat(1, audio_len, 1, 1)
        memory_tiled = memory_tiled.view(-1, embed_dim) # (batch_size * audio_len * text_len, embed_dim)

        info_matrix = torch.tanh(self.W_q(query_tiled) + self.W_c(memory_tiled))

        v_tiled = self.v.unsqueeze(0).repeat(batch_size * audio_len * text_len, 1)

        energy = torch.sum(v_tiled * info_matrix, dim=1)
        energy = energy.view(batch_size, audio_len, text_len) # (batch_size, audio_len, text_len)
        energy = energy.float().masked_fill(memory_mask, float('-inf')).type_as(energy)

        alignment = F.softmax(energy.float(), dim=2).type_as(energy)  # (batch_size, audio_len, text_len)

        return alignment

