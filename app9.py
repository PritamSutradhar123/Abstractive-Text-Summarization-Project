from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

def generate2_summary2(input_text):
    # Load the tokenizer
    with open('s_tokenizer.pkl', 'rb') as f:
        s_tokenizer = pickle.load(f)

    # Load the model
    enc_model = tf.keras.models.load_model('encoder_model.h5')
    dec_model = tf.keras.models.load_model('decoder_model.h5')

    # Tokenize the input text
    input_seq = s_tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=800, padding='post')

    # Generate the summary
    h, c = enc_model.predict(input_seq)

    next_token = np.zeros((1, 1))
    next_token[0, 0] = s_tokenizer.word_index['sostok']
    output_seq = ''

    stop = False
    count = 0

    while not stop:
        if count > 100:
            break
        decoder_out, state_h, state_c = dec_model.predict([next_token]+[h, c])
        token_idx = np.argmax(decoder_out[0, -1, :])

        if token_idx == s_tokenizer.word_index['eostok']:
            stop = True
        elif token_idx > 0 and token_idx != s_tokenizer.word_index['sostok']:
            token = s_tokenizer.index_word[token_idx]
            output_seq = output_seq + ' ' + token

        next_token = np.zeros((1, 1))
        next_token[0, 0] = token_idx
        h, c = state_h, state_c
        count += 1

    return output_seq.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        get_sum_for_text = generate2_summary2(text)
        return render_template('index9.html', summary=get_sum_for_text, text=text)
    return render_template('index9.html')

if __name__ == '__main__':
    app.run()