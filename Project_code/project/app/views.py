from django.shortcuts import render
from django.utils.datastructures import MultiValueDictKeyError
import cv2
from gtts import gTTS
import os
from django import forms
from .models import *

from django.shortcuts import render, redirect
from .models import *
 
import numpy as np


import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

import shutil
import tqdm
import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import json
import random
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from tensorflow.keras.models import Model
from keras.utils import to_categorical
import model
import functools
import operator
import joblib
import shutil
import tqdm
import cv2
import time


test_path = "testing_data/video"

def video_to_frames(video):
    path = 'temporary_images'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    video_path = os.path.join(test_path, video)
    count = 0
    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cv2.imwrite(os.path.join('temporary_images/', 'frame%d.jpg' % count), frame)
        image_list.append(os.path.join('temporary_images/', 'frame%d.jpg' % count))
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return image_list


def model_cnn_load():
    model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    out = model.layers[-2].output
    model_final = Model(inputs=model.input, outputs=out)
    return model_final


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img


def extract_features(video, model):
    video_id = video.split(".")[0]
    print(video_id)
    print(f'Processing video {video}')
    try:
        image_list = video_to_frames(video)
        samples = np.round(np.linspace(
            0, len(image_list) - 1, 80))
        print("Images List :" , image_list)
        image_list = [image_list[int(sample)] for sample in samples]
        images = np.zeros((len(image_list), 224, 224, 3))
        for i in range(len(image_list)):
            img = load_image(image_list[i])
            images[i] = img
        images = np.array(images)
        fc_feats = model.predict(images, batch_size=128)
        img_feats = np.array(fc_feats)
        # cleanup
        shutil.rmtree('temporary_images')
        return img_feats
    except Exception as e:
        print(e)


def extract_feats_pretrained_cnn():
    model = model_cnn_load()
    print('Model loaded')

    if not os.path.isdir(os.path.join(test_path, 'feat')):
        os.mkdir(os.path.join(test_path, 'feat'))

    video_list = os.listdir(os.path.join(test_path, 'video'))
    
    #ًWhen running the script on Colab an item called '.ipynb_checkpoints' 
    #is added to the beginning of the list causing errors later on, so the next line removes it.
    #video_list.remove('.ipynb_checkpoints')
    
    for video in video_list:

        outfile = os.path.join(test_path, 'feat', video + '.npy')
        img_feats = extract_features(video, model)
        np.save(outfile, img_feats)





class VideoDescriptionRealTime(object):
    """
        Initialize the parameters for the model
        """
    def __init__(self):
        self.latent_dim = 512
        self.num_encoder_tokens = 4096
        self.num_decoder_tokens = 1500
        self.time_steps_encoder = 80
        self.max_probability = -1

        # models
        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = model.inference_model()
        
        #self.inf_decoder_model = None
        self.save_model_path = 'model_final'
        test_path = "testing_data/video"
        self.search_type = "greedy"
        self.num = 0


    def greedy_search(self, loaded_array):
        """

        :param f: the loaded numpy array after creating videos to frames and extracting features
        :return: the final sentence which has been predicted greedily
        """
        inv_map = self.index_to_word()
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, 1500))
        final_sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)
            if y_hat == 0:
                continue
            if inv_map[y_hat] is None:
                break
            if inv_map[y_hat] == 'eos':
                break
            else:
                final_sentence = final_sentence + inv_map[y_hat] + ' '
                target_seq = np.zeros((1, 1, 1500))
                target_seq[0, 0, y_hat] = 1
        return final_sentence

    def decode_sequence2bs(self, input_seq):
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value, [], [], 0)
        return decode_seq
  

    def decoded_sentence_tuning(self, decoded_sentence):
        # tuning sentence
        decode_str = []
        filter_string = ['bos', 'eos']
        uni_gram = {}
        last_string = ""
        for idx2, c in enumerate(decoded_sentence):
            if c in uni_gram:
                uni_gram[c] += 1
            else:
                uni_gram[c] = 1
            if last_string == c and idx2 > 0:
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if idx2 > 0:
                last_string = c
        return decode_str

    def index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word

    def get_test_data(self):
        # loads the features array
        file_list = os.listdir(test_path)
        #print(file_list)
        try:
            file_list.remove('.DS_Store')
        except Exception as e:
            print(e)
        #print(file_list)
        # with open(os.path.join(test_path, 'testing.txt')) as testing_file:
            # lines = testing_file.readlines()
        # file_name = lines[self.num].strip()
        file_name = file_list[self.num]
        path = os.path.join(test_path, 'feat', file_name + '.npy')
        if os.path.exists(path):
            f = np.load(path)
        else:
            model = model_cnn_load()
            f = extract_features(file_name, model)
        if self.num < len(file_list):
            self.num += 1
        else:
            self.num = 0
        return f, file_name

    def test(self):
        X_test, filename = self.get_test_data()
        #print(X_test)
        # generate inference test outputs
        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(X_test.reshape((-1, 80, 4096)))
        else:
            sentence_predicted = ''
            decoded_sentence = self.decode_sequence2bs(X_test.reshape((-1, 80, 4096)))
            decode_str = self.decoded_sentence_tuning(decoded_sentence)
            for d in decode_str:
                sentence_predicted = sentence_predicted + d + ' '
        # re-init max prob
        self.max_probability = -1
        return sentence_predicted, filename

    def main(self, filename, caption):
        """

        :param filename: the video to load
        :param caption: final caption
        :return:
        """
        try:
            # 1. Initialize reading video object
            cap1 = cv2.VideoCapture(os.path.join(test_path, filename))
            cap2 = cv2.VideoCapture(os.path.join(test_path, filename))
            caption = '[' + ' '.join(caption.split()[1:]) + ']'
            # 2. Cycle through pictures
            while cap1.isOpened():
                ret, frame = cap2.read()
                ret2, frame2 = cap1.read()
                if ret:
                    imS = cv2.resize(frame, (480, 300))
                    cv2.putText(imS, caption, (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2, cv2.LINE_4)
                    cv2.imshow("VIDEO CAPTIONING", imS)
                if ret2:
                    imS = cv2.resize(frame, (480, 300))
                    cv2.imshow("ORIGINAL", imS)
                else:
                    break

                # Quit playing
                key = cv2.waitKey(25)
                if key == 27:  # Button esc
                    break

            # 3. Free resources
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)




# Create your views here.
def index(request):
    if(request.method=="POST"): 
        # # subject=request.POST['img']
        # img = request.POST['img']
        # print("jaya")
        # answer="sodhi jaya"
        # print(answer)


        folder_path = 'testing_data/video'
        
        files = os.listdir(folder_path)

        # loop through the list of files and remove them one by one
        for file in files:
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
        
        
        
          # replace with the path to the folder you want to store the file in
        uploaded_file = request.FILES['img']  # replace 'file' with the name of the file field in your form

        # create the destination folder if it doesn't already exist
        
        destination_path = os.path.join(folder_path, uploaded_file.name)

        # save the uploaded file to the destination folder
        with open(destination_path, 'wb+') as destination_file:
            for chunk in uploaded_file.chunks():
                destination_file.write(chunk)

        video_to_text = VideoDescriptionRealTime()
        start = time.time()
        video_caption, file = video_to_text.test()
        end = time.time()
        sentence = ''
        print(sentence)
        for text in video_caption.split():
            sentence = sentence + ' ' + text
        
        h2=request.FILES['img']
        h1 = Hotel.objects.create(hotel_Main_Img=h2)
        
        h1.save()
        p1='static/'
        path=str(p1)+str(h2)
        
        # mytext = "hello jaya mam"
        audio = gTTS(text=sentence, lang="en", slow=False)
        audio.save("static/example6.mp3")
        
        print(path)
        
        return render(request,"index.html",{'image_path':path,'str':sentence})
    else:
         return render(request,"index.html",{'message':'Invalid Credentials'})