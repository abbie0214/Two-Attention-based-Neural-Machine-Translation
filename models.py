# models.py

import torch
import torch.nn as nn
from utils import *


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size


        self.W_xr = nn.Linear(self.input_size,self.hidden_size,  False)
        self.W_hr = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_xz = nn.Linear(self.input_size,self.hidden_size,  False)
        self.W_hz = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_xg = nn.Linear(self.input_size,self.hidden_size,  False)
        self.W_hg = nn.Linear(self.hidden_size, self.hidden_size)



    def forward(self, x, h_prev):

        r = torch.sigmoid(self.W_xr(x) + self.W_hr(h_prev))
        z = torch.sigmoid(self.W_xz(x) + self.W_hz(h_prev))
        g = torch.tanh(self.W_xg(x) +  r * self.W_hg(h_prev))
        h_new = (1 - z) * g + z * h_prev

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

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)  
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]  
            hidden = self.gru(x, hidden)
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden

    def init_hidden(self, bs):
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
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  

        hiddens = []
        h_prev = hidden_init
        for i in range(seq_len):
            x = embed[:,i,:] 
            h_prev = self.rnn(x, h_prev)  
            hiddens.append(h_prev)

        hiddens = torch.stack(hiddens, dim=1) 
        
        output = self.out(hiddens)  
        return output, None


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()

        self.hidden_size = hidden_size.
       	self.attention_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )


        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, keys, values):

        batch_size = keys.size(0)

        expanded_queries = queries.unsqueeze(1).expand_as(keys)
        concat_inputs = torch.cat((expanded_queries, keys), dim = 2)
        unnormalized_attention = self.attention_network(concat_inputs)
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.sum(torch.mul(attention_weights, values),dim = 1, keepdim = True)

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
  
        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        hiddens = []
        attentions = []
        h_prev = hidden_init
        for i in range(seq_len):
        
            embed_current = embed[:,i,:]
            context, attention_weights = self.attention(h_prev, annotations, annotations)
            embed_and_context = torch.cat((embed_current, context[:,0,:]), dim = 1)
            h_prev = self.rnn(embed_and_context, h_prev)


            hiddens.append(h_prev)
            attentions.append(attention_weights)

        hiddens = torch.stack(hiddens, dim=1) 
        attentions = torch.cat(attentions, dim=2) 
        
        output = self.out(hiddens) 
        return output, attentions


class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=2) 
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
 
        batch_size = queries.size(0)
        if queries.dim() != 3:
            queries = torch.unsqueeze(queries, dim = 1)
        q = self.Q(queries)
        k = self.K(keys)
        k = torch.transpose(k, 1, 2)
        v = self.V(values)
        unnormalized_attention = torch.bmm(q,k) * self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights, v)
        return context, attention_weights


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
   
        batch_size = queries.size(0)

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
        context = torch.bmm(attention_weights, v)

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
       
        batch_size, seq_len = inputs.size()

        
        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        encoded = encoded + self.positional_encodings[:seq_len]

        annotations = encoded

        for i in range(self.num_layers):
 
          new_annotations, self_attention_weights = self.self_attentions[i](annotations,annotations,annotations)
          residual_annotations = annotations + new_annotations
          new_annotations = self.attention_mlps[i](residual_annotations.view(-1, self.hidden_size)).view(batch_size, seq_len, self.hidden_size)
          annotations = residual_annotations + new_annotations

        return annotations, None  

    def create_positional_encodings(self, max_seq_len=1000):

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
        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        embed = embed + self.positional_encodings[:seq_len]       

        encoder_attention_weights_list = []
        self_attention_weights_list = []
        contexts = embed
        for i in range(self.num_layers):

          new_contexts, self_attention_weights = self.self_attentions[i](contexts,contexts,contexts)
          residual_contexts = contexts + new_contexts

          new_contexts, encoder_attention_weights = self.encoder_attentions[i](residual_contexts,annotations,annotations)
          residual_contexts = residual_contexts + new_contexts
          new_contexts = self.attention_mlps[i](residual_contexts.view(-1, self.hidden_size)).view(batch_size, seq_len, self.hidden_size)
          contexts = residual_contexts + new_contexts

          
          encoder_attention_weights_list.append(encoder_attention_weights)
          self_attention_weights_list.append(self_attention_weights)
          
        output = self.out(contexts)
        encoder_attention_weights = torch.stack(encoder_attention_weights_list)
        self_attention_weights = torch.stack(self_attention_weights_list)
        
        return output, (encoder_attention_weights, self_attention_weights)

    def create_positional_encodings(self, max_seq_len=1000):
    
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
