#!/usr/bin/env python
# coding: utf-8

# In[16]:



from tensorflow.keras import preprocessing , utils , layers, activations , models
import os
import yaml
import numpy as np
import tensorflow as tf
import pickle
#import matplotlib.pylot as plt

dir_path = 'chatbot_nlp/data'
files_list = os.listdir(dir_path + os.sep)

questions = list()
answers = list()
n_conv = 0

for filepath in files_list:
    stream = open( dir_path + os.sep + filepath , 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        n_conv = n_conv + 1
        if len( con ) > 2 :
            questions.append(con[0])
            replies = con[ 1 : ]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append( ans )
        elif len( con )> 1:
            questions.append(con[0])
            answers.append(con[1])

# Remove any question whose answer is not a str
answers_with_tags = list()
for i in range( len( answers ) ):
    if type( answers[i] ) == str:
        answers_with_tags.append( answers[i] )
    else:
        questions.pop( i )

# add tags <START> and <END> to every answer        
answers = list()
for i in range( len( answers_with_tags ) ) :
    answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )

# Define tokenizer 
tokenizer = preprocessing.text.Tokenizer()

# Updating internal vocabulary of tokenizer from answer+questions
tokenizer.fit_on_texts( questions + answers )

VOCAB_SIZE = len( tokenizer.word_index )+1
print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))

#Checking numpy version for debugging
print("Number of conversations: ",n_conv)


# In[2]:



from gensim.models import Word2Vec
import re


vocab = []
for word in tokenizer.word_index:
    vocab.append( word )

def tokenize( sentences ):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub( '[^a-zA-Z]', ' ', sentence )
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append( tokens )
    return tokens_list , vocabulary

#p = tokenize( questions + answers )
#model = Word2Vec( p[ 0 ] ) 

#embedding_matrix = np.zeros( ( VOCAB_SIZE , 100 ) )
#for i in range( len( tokenizer.word_index ) ):
    #embedding_matrix[ i ] = model[ vocab[i] ]

# encoder_input_data

#Transforms each text in texts to a sequence of integers.
tokenized_questions = tokenizer.texts_to_sequences( questions )

# Determining maximum length of a question for padding purposes
maxlen_questions = max( [ len(x) for x in tokenized_questions ] )

# Padding the sequences
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions , maxlen=maxlen_questions , padding='post' )

#Checking how it looks
#for i in range(5):
 #   print(padded_questions[i])

# converting the padded sequences into an np array
encoder_input_data = np.array( padded_questions )
print("Encoder input data shape and maxlen:")
print( encoder_input_data.shape , maxlen_questions )

print("\nEncoder input data:")
print(encoder_input_data)

# decoder_input_data
#Transforms each text in texts to a sequence of integers.
tokenized_answers = tokenizer.texts_to_sequences( answers )

# Determining maximum length of a question for padding purposes
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )

#Padding them
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )
print("\nDecoder input data shape and maxlen:")
print( decoder_input_data.shape , maxlen_answers )
print("\nDecoder input data (np array):")
print(decoder_input_data)


# decoder_output_data

#Transforms each text in texts to a sequence of integers.
tokenized_answers = tokenizer.texts_to_sequences( answers )

# Removing the <START> tag from every answer
for i in range(len(tokenized_answers)) :
    tokenized_answers[i] = tokenized_answers[i][1:]

# Padding the sequences
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )

# One hot encoding the answers
onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE ) # number of classes = VOCAB_SIZE
decoder_output_data = np.array( onehot_answers )

# Logging
print("Decoder output data shape:")
print( decoder_output_data.shape )
print("\nDecoder output data:")
print(decoder_output_data)


# In[3]:


#Model architecture
# general idea is that:
#        output_tensor = layer(input_tensor)

# We have input layers for encoder and decoder input data

# Then we have embedding layers for them

# for encoder, we have LSTM layer which gives us encoder_outputs, and states. States are important? They are weights values

encoder_inputs = tf.keras.layers.Input(shape=( maxlen_questions , ))
encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

# for decoder, we have LSTM layer with no input tensor but initial states are from encoder! 

decoder_inputs = tf.keras.layers.Input(shape=( maxlen_answers ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)

# Why return_sequences = true?   What is decoder_lstm? is is similar to encoder_outputs from above LSTM layer.
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )

# Just your regular densely-connected NN layer.
# Dense implements the operation: output = activation(dot(input, kernel) + bias) 
# where activation is the element-wise activation function passed as the activation argument, 
# kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer 
# (only applicable if use_bias is True). These are all attributes of Dense.

decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()


# In[4]:


# Training and saving the model

#try:
 #   model = tf.keras.models.load_model('model.h5')
  #  print("Model Successfully loaded.")
#except Exception as inst:
 #   print(inst)
history = model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150 ) 
model.save( 'model.h5' )


# In[5]:


# We create inference models which help in predicting answers.

# Encoder inference model : Takes the question as input and outputs LSTM states ( h and c ).

# Decoder inference model : Takes in 2 inputs, one are the LSTM states ( Output of encoder model ), 

# second are the answer input seqeunces ( ones not having the <start> tag ). 

# It will output the answers for the question which we fed to the encoder model and its state values.

def make_inference_models():
    
    # encoder_inputs are the questions and encoder_states are well.. the LSTM states from encoder_iput_data stream.
    # Model groups layers into an object with training and inference features.
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    # Defining the state tensors 
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # We are using the decoder_lstm() again?
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs) #decoder_embedding are the embedded answers with <START>
   
    # Obtain decoder_states...
    decoder_states = [state_h, state_c]
    
    # decoder_dense is a Dense layer (VOCAB_SIZE, softmax)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


# In[6]:


# Convert str questions to integer tokens with padding. 

def str_to_tokens( sentence : str ):
    sentence = sentence.lower()
    sentence = re.sub( '[^a-zA-Z]', ' ', sentence )
    words = sentence.split()
    tokens_list = list()
    #tokens_list = tokenizer.texts_to_sequences( sentence )
    # Appending the associated token of the word in tokenizer to the tokens_list 
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
        
        #Pad the tokens_list to the max question length (22)
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')


# In[7]:


# 1. First, we take a question as input and predict the state values using enc_model.
# 2. We set the state values in the decoder's LSTM.
# 3. Then, we generate a sequence which contains the <start> element.
# 4. We input this sequence in the dec_model.
# 5. We replace the <start> element with the element which was predicted by the dec_model and update the state values.
# 6. We carry out the above steps iteratively till we hit the <end> tag or the maximum answer length.


enc_model , dec_model = make_inference_models()


# In[ ]:


while True:
    
    # Predicting state values using the enc_model on the input question.
    q_str = input( 'Enter question : ' )
    
    if q_str.lower() == 'end':
        break
    try:
        states_values = enc_model.predict( str_to_tokens( q_str ) )
    except KeyError as err:
        print(err)
        print('My vocabulary is limited for now, please try again')
        continue
    # Generating a sequence which contains the <start> element.
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    
    
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition :
        
        # Setting values obtained from enc_model in the decoder's LSTM, also input the empty sequence with <start> in it.
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        
        # Returns the indices of the maximum values along the axis
        # Why this line??
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        
        # Iterate through tokenizer's word index to find the word corresponding to sampled_word_index and 
        # add it to the decoded translation
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                if word != 'end':
                    decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        # Stop loop if reached end or length of answer is bigger than max len for answers. 
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
        
        # replace <start> with predicted element and save states. 
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 
    
    
    #decoded_translation.replace("end","bye")
        

    print( decoded_translation )

