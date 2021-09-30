# main.py


from train import train
from utils import visualize_attention, translate_sentence, AttrDict, print_opts


"""
Train the RNN language model comprised of recurrent encoder and decoders. 
"""

TEST_SENTENCE = 'the air conditioning is working'

args = AttrDict()
args_dict = {
              'cuda':False, 
              'nepochs':100, 
              'checkpoint_dir':"checkpoints", 
              'learning_rate':0.005, 
              'lr_decay':0.99,
              'batch_size':64, 
              'hidden_size':20, 
              'encoder_type': 'rnn', # options: rnn / transformer
              'decoder_type': 'rnn', # options: rnn / rnn_attention / transformer
              'attention_type': '',  # options: additive / scaled_dot
}
args.update(args_dict)

print_opts(args)
rnn_encoder, rnn_decoder = train(args)

translated = translate_sentence(TEST_SENTENCE, rnn_encoder, rnn_decoder, None, args)
print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))

"""Try translating different sentences by changing the variable TEST_SENTENCE. Identify two distinct failure modes and briefly describe them."""

TEST_SENTENCE = 'the air conditioning is working'
translated = translate_sentence(TEST_SENTENCE, rnn_encoder, rnn_decoder, None, args)
print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))

"""
Train the RNN language model comprised of recurrent encoder and decoders with the additive attention component. 
"""

TEST_SENTENCE = 'the air conditioning is working'

args = AttrDict()
args_dict = {
              'cuda':False, 
              'nepochs':100, 
              'checkpoint_dir':"checkpoints", 
              'learning_rate':0.005, 
              'lr_decay':0.99,
              'batch_size':64, 
              'hidden_size':20, 
              'encoder_type': 'rnn', # options: rnn / transformer
              'decoder_type': 'rnn_attention', # options: rnn / rnn_attention / transformer
              'attention_type': 'additive',  # options: additive / scaled_dot
}
args.update(args_dict)

print_opts(args)
rnn_attn_encoder, rnn_attn_decoder = train(args)

translated = translate_sentence(TEST_SENTENCE, rnn_attn_encoder, rnn_attn_decoder, None, args)
print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))

"""Try translating different sentences by changing the variable TEST_SENTENCE. Identify two distinct failure modes and briefly describe them."""

TEST_SENTENCE = 'the air conditioning is working'
translated = translate_sentence(TEST_SENTENCE, rnn_attn_encoder, rnn_attn_decoder, None, args)
print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))

"""
Train the Transformer language model comprised of a (simplified) transformer encoder and transformer decoder.
"""

TEST_SENTENCE = 'the air conditioning is working'

args = AttrDict()
args_dict = {
              'cuda':False, 
              'nepochs':100, 
              'checkpoint_dir':"checkpoints", 
              'learning_rate':0.005, ## INCREASE BY AN ORDER OF MAGNITUDE
              'lr_decay':0.99,
              'batch_size':64, 
              'hidden_size':20, 
              'encoder_type': 'transformer',
              'decoder_type': 'transformer', # options: rnn / rnn_attention / transformer
              'num_transformer_layers': 3,
}
args.update(args_dict)

print_opts(args)
transformer_encoder, transformer_decoder = train(args)

translated = translate_sentence(TEST_SENTENCE, transformer_encoder, transformer_decoder, None, args)
print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))

"""Try translating different sentences by changing the variable TEST_SENTENCE. Identify two distinct failure modes and briefly describe them."""

TEST_SENTENCE = 'the air conditioning is working'
translated = translate_sentence(TEST_SENTENCE, transformer_encoder, transformer_decoder, None, args)
print("source:\t\t{} \ntranslated:\t{}".format(TEST_SENTENCE, translated))

"""# Attention Visualizations

One of the benefits of using attention is that it allows us to gain insight into the inner workings of the model.

By visualizing the attention weights generated for the input tokens in each decoder step, we can see where the model focuses while producing each output token.

The code in this section loads the model you trained from the previous section and uses it to translate a given set of words: it prints the translations and display heatmaps to show how attention is used at each step.

Play around with visualizing attention maps generated by the previous two models you've trained. Inspect visualizations in one success and one failure case for both models.
"""

TEST_WORD_ATTN = 'street'
visualize_attention(TEST_WORD_ATTN, rnn_attn_encoder, rnn_attn_decoder, None, args, save="additive_attention.pdf")

TEST_WORD_ATTN = 'street'
visualize_attention(TEST_WORD_ATTN, transformer_encoder, transformer_decoder, None, args, save="scaled_dot_product_attention.pdf")