# models.py

import torch
import torch.nn as nn
from utils import *


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size


        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------

        # self.Wr = nn.Linear(...)
        self.W_xr = nn.Linear(self.input_size,self.hidden_size,  False)
        self.W_hr = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_xz = nn.Linear(self.input_size,self.hidden_size,  False)
        self.W_hz = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_xg = nn.Linear(self.input_size,self.hidden_size,  False)
        self.W_hg = nn.Linear(self.hidden_size, self.hidden_size)


        # ------------
        # FILL THIS IN - END
        # ------------
        

    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """

        # ------------
        # FILL THIS IN - START
        # ------------

        # z = ...
        # r = ...
        # g = ...
        # h_new = ...

        r = torch.sigmoid(self.W_xr(x) + self.W_hr(h_prev))
        z = torch.sigmoid(self.W_xz(x) + self.W_hz(h_prev))
        g = torch.tanh(self.W_xg(x) +  r * self.W_hg(h_prev))
        h_new = (1 - z) * g + z * h_prev

        # ------------
        # FILL THIS IN - END
        # ------------
        return h_new


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, opts):
        super(GRUEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.opts = opts

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = MyGRUCell(hidden_size, hidden_size)

    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru(x, hidden)
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden

    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. (batch_size x hidden_size)
        """
        return to_var(torch.zeros(bs, self.hidden_size), self.opts.cuda)

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = MyGRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, annotations, hidden_init):
        """Forward pass of the non-attentional decoder RNN.

        Arguments:
            inputs: Input token indexes across a batch. (batch_size x seq_len)
            annotations: This is not used here. It just maintains consistency with the
                    interface used by the AttentionDecoder class.
            hidden_init: The hidden states from the last step of encoder, across a batch. (batch_size x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            None        
        """        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size        

        hiddens = []
        h_prev = hidden_init
        for i in range(seq_len):
            x = embed[:,i,:]  # Get the current time step input tokens, across the whole batch
            h_prev = self.rnn(x, h_prev)  # batch_size x hidden_size
            hiddens.append(h_prev)

        hiddens = torch.stack(hiddens, dim=1) # batch_size x seq_len x hidden_size
        
        output = self.out(hiddens)  # batch_size x seq_len x vocab_size
        return output, None


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()

        self.hidden_size = hidden_size

        # Create a two layer fully-connected network. Hint: Use nn.Sequential
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1

        # ------------
        # FILL THIS IN - START
        # ------------

        # self.attention_network = ...
       	self.attention_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # ------------
        # FILL THIS IN - END
        # ------------

        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, keys, values):
        """The forward pass of the additive attention mechanism.

        Arguments:
            queries: The current decoder hidden state. (batch_size x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x 1 x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The attention_weights must be a softmax weighting over the seq_len annotations.
        """

        batch_size = keys.size(0)


        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------
        
        # expanded_queries = ...
        # concat_inputs = ...
        # unnormalized_attention = self.attention_network(...)
        # attention_weights = self.softmax(...)
        # context = ...

        expanded_queries = queries.unsqueeze(1).expand_as(keys)
        concat_inputs = torch.cat((expanded_queries, keys), dim = 2)
        unnormalized_attention = self.attention_network(concat_inputs)
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.sum(torch.mul(attention_weights, values),dim = 1, keepdim = True)


        # ------------
        # FILL THIS IN - END
        # ------------

        return context, attention_weights


class RNNAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, attention_type='scaled_dot'):
        super(RNNAttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = MyGRUCell(input_size=hidden_size*2, hidden_size=hidden_size)
        if attention_type == 'additive':
          self.attention = AdditiveAttention(hidden_size=hidden_size)
        elif attention_type == 'scaled_dot':
          self.attention = ScaledDotAttention(hidden_size=hidden_size)
        
        self.out = nn.Linear(hidden_size, vocab_size)

        
    def forward(self, inputs, annotations, hidden_init):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: The final hidden states from the encoder, across a batch. (batch_size x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """
        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        hiddens = []
        attentions = []
        h_prev = hidden_init
        for i in range(seq_len):
            # You are free to follow the code template below, or do it a different way,
            # as long as the output is correct.

            # ------------
            # FILL THIS IN - START
            # ------------

            # embed_current = ...  # Get the current time step, across the whole batch
            # context, attention_weights = self.attention(...)  # batch_size x 1 x hidden_size
            # embed_and_context = ....  # batch_size x (2*hidden_size)
            # h_prev = self.rnn(...)  # batch_size x hidden_size


            embed_current = embed[:,i,:]
            context, attention_weights = self.attention(h_prev, annotations, annotations)
            embed_and_context = torch.cat((embed_current, context[:,0,:]), dim = 1)
            h_prev = self.rnn(embed_and_context, h_prev)

            
            # ------------
            # FILL THIS IN - END
            # ------------
            
            hiddens.append(h_prev)
            attentions.append(attention_weights)

        hiddens = torch.stack(hiddens, dim=1) # batch_size x seq_len x hidden_size
        attentions = torch.cat(attentions, dim=2) # batch_size x seq_len x seq_len
        
        output = self.out(hiddens)  # batch_size x seq_len x vocab_size
        return output, attentions


class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=2) #!! dim = 2
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x k x seq_len)

            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size = queries.size(0)

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        # ------------
        # FILL THIS IN - START
        # ------------
        
        # if queries.dim() != 3:
        #     queries = ...
        # q = ...
        # k = ...
        # v = ...
        # unnormalized_attention = ...
        # attention_weights = ...
        # context = ...


        if queries.dim() != 3:
            queries = torch.unsqueeze(queries, dim = 1)
        q = self.Q(queries)
        # print('q.shape',q.shape)
        k = self.K(keys)
        # print('k1.shape',k.shape)
        k = torch.transpose(k, 1, 2)
        # print('k2.shape', k.shape)
        v = self.V(values)
        unnormalized_attention = torch.bmm(q,k) * self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights, v)

        # ------------
        # FILL THIS IN - END
        # ------------

        return context, attention_weights

"""## Step 2: Implement Causal Dot-Product Attention
Now implement the scaled causal dot product described in the assignment worksheet. 
"""

class CausalScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CausalScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size
        self.neg_inf = -1e7

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x k)

            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size = queries.size(0)


        # ------------
        # FILL THIS IN - START
        # ------------
        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.
        
        # if queries.dim() != 3:
        #     queries = ...
        # q = ...
        # k = ...
        # v = ...
        # unnormalized_attention = ....
        # mask = ... 
        # attention_weights = ...
        # context = ...

        k_value = queries.size(1)
        seq_len = keys.size(1)

        if queries.dim() != 3:
            queries = torch.unsqueeze(queries, dim = 1)
        q = self.Q(queries)
        k = self.K(keys)
        k = torch.transpose(k, 1, 2)
        v = self.V(values)
        unnormalized_attention = torch.bmm(q,k) * self.scaling_factor
        mask = torch.tril(torch.ones(batch_size, k_value, seq_len))
        unnormalized_attention[mask == 0] = self.neg_inf
        attention_weights = self.softmax(unnormalized_attention)
        # print('attention_weights.shape',attention_weights.shape)
        context = torch.bmm(attention_weights, v)
        # ------------
        # FILL THIS IN - END
        # ------------
        return context, attention_weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, opts):
        super(TransformerEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.opts = opts

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.self_attentions = nn.ModuleList([ScaledDotAttention(
                                    hidden_size=hidden_size, 
                                 ) for i in range(self.num_layers)])

        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                 ) for i in range(self.num_layers)])

        self.positional_encodings = self.create_positional_encodings()

    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        # ------------
        # FILL THIS IN - START
        # ------------
        
        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        # Add positional embeddings from self.create_positional_encodings. (a'la https://arxiv.org/pdf/1706.03762.pdf, section 3.5)
        # encoded = ...

        encoded = encoded + self.positional_encodings[:seq_len]

        annotations = encoded

        for i in range(self.num_layers):
          # new_annotations, self_attention_weights = ...  # batch_size x seq_len x hidden_size
          new_annotations, self_attention_weights = self.self_attentions[i](annotations,annotations,annotations)
          residual_annotations = annotations + new_annotations
          new_annotations = self.attention_mlps[i](residual_annotations.view(-1, self.hidden_size)).view(batch_size, seq_len, self.hidden_size)
          annotations = residual_annotations + new_annotations
        # ------------
        # FILL THIS IN - END
        # ------------

        # Transformer encoder does not have a last hidden layer. 
        return annotations, None  

    def create_positional_encodings(self, max_seq_len=1000):
      """Creates positional encodings for the inputs.

      Arguments:
          max_seq_len: a number larger than the maximum string length we expect to encounter during training

      Returns:
          pos_encodings: (max_seq_len, hidden_dim) Positional encodings for a sequence with length max_seq_len. 
      """
      pos_indices = torch.arange(max_seq_len)[..., None]
      dim_indices = torch.arange(self.hidden_size//2)[None, ...]
      exponents = (2*dim_indices).float()/(self.hidden_size)
      trig_args = pos_indices / (10000**exponents)
      sin_terms = torch.sin(trig_args)
      cos_terms = torch.cos(trig_args)

      pos_encodings = torch.zeros((max_seq_len, self.hidden_size))
      pos_encodings[:, 0::2] = sin_terms
      pos_encodings[:, 1::2] = cos_terms

      if self.opts.cuda:
        pos_encodings = pos_encodings.cuda()

      return pos_encodings

"""## Step 4: Transformer Decoder
Complete the following transformer decoder implementation. 
"""

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, is_cuda):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.is_cuda = is_cuda

        self.embedding = nn.Embedding(vocab_size, hidden_size)        
        self.num_layers = num_layers

        self.self_attentions = nn.ModuleList([CausalScaledDotAttention(
                                    hidden_size=hidden_size,
                                 ) for i in range(self.num_layers)])

        # self.self_attentions = nn.ModuleList([ScaledDotAttention(
        #                             hidden_size=hidden_size,
        #                          ) for i in range(self.num_layers)])

        self.encoder_attentions = nn.ModuleList([ScaledDotAttention(
                                    hidden_size=hidden_size, 
                                 ) for i in range(self.num_layers)])

        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                 ) for i in range(self.num_layers)])
        self.out = nn.Linear(hidden_size, vocab_size)

        self.positional_encodings = self.create_positional_encodings()

    def forward(self, inputs, annotations, hidden_init):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: Not used in the transformer decoder
        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """
        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        embed = embed + self.positional_encodings[:seq_len]       

        encoder_attention_weights_list = []
        self_attention_weights_list = []
        contexts = embed
        for i in range(self.num_layers):
          # ------------
          # FILL THIS IN - START
          # ------------
          # new_contexts, self_attention_weights = ... # batch_size x seq_len x hidden_size
          new_contexts, self_attention_weights = self.self_attentions[i](contexts,contexts,contexts)
          residual_contexts = contexts + new_contexts
          # new_contexts, encoder_attention_weights = ...  # batch_size x seq_len x hidden_size
          new_contexts, encoder_attention_weights = self.encoder_attentions[i](residual_contexts,annotations,annotations)
          residual_contexts = residual_contexts + new_contexts
          new_contexts = self.attention_mlps[i](residual_contexts.view(-1, self.hidden_size)).view(batch_size, seq_len, self.hidden_size)
          contexts = residual_contexts + new_contexts
          # ------------
          # FILL THIS IN - END
          # ------------
          
          encoder_attention_weights_list.append(encoder_attention_weights)
          self_attention_weights_list.append(self_attention_weights)
          
        output = self.out(contexts)
        encoder_attention_weights = torch.stack(encoder_attention_weights_list)
        self_attention_weights = torch.stack(self_attention_weights_list)
        
        return output, (encoder_attention_weights, self_attention_weights)

    def create_positional_encodings(self, max_seq_len=1000):
      """Creates positional encodings for the inputs.

      Arguments:
          max_seq_len: a number larger than the maximum string length we expect to encounter during training

      Returns:
          pos_encodings: (max_seq_len, hidden_dim) Positional encodings for a sequence with length max_seq_len. 
      """
      pos_indices = torch.arange(max_seq_len)[..., None]
      dim_indices = torch.arange(self.hidden_size//2)[None, ...]
      exponents = (2*dim_indices).float()/(self.hidden_size)
      trig_args = pos_indices / (10000**exponents)
      sin_terms = torch.sin(trig_args)
      cos_terms = torch.cos(trig_args)

      pos_encodings = torch.zeros((max_seq_len, self.hidden_size))
      pos_encodings[:, 0::2] = sin_terms
      pos_encodings[:, 1::2] = cos_terms

      if self.is_cuda == True:
          pos_encodings = pos_encodings.cuda()

      return pos_encodings
