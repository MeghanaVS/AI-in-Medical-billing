#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install SpeechRecognition')
get_ipython().system('pip install moviepy')
get_ipython().system('pip install moviepy pydub')


# In[4]:


get_ipython().system('pip install pytube')


# In[5]:


get_ipython().system('pip install openai==1.1.0')
get_ipython().system('pip install python-docx')
get_ipython().system('pip install torch')
get_ipython().system('pip install transformers')


# In[6]:


import openai
print(openai.__version__)


# In[3]:


import os
import re
import pydub
import openai
import moviepy.editor as mpe
import torch
import speech_recognition as sr
import pytube

#from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
#from pytube import YouTube
#from moviepy.editor import VideoFileClip


# In[36]:


# Output path for downloaded audio
output_path = "/Users/meghanavs/Desktop/"

audio_file_path = os.path.join(output_path, "ICD.mp3")


# In[37]:


# Specify the audio file path
path = audio_file_path

# Transcribe the audio
client = OpenAI(
    api_key="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",
)
response = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(path, 'rb'),
    response_format="text"
)

# Extract the transcription
print(response)

transcription = response.choices[0].text if hasattr(response, 'choices') and response.choices else None

if transcription is not None:
    # Generate meeting minutes
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-snli")
    encoded_text = tokenizer(transcription, truncation=True, padding=True, return_tensors='pt')

    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-snli")

    with torch.no_grad():
        output = model(**encoded_text)
        logits = output.logits
        predictions = logits.argmax(-1)

    # Check if the transcription is classified as meeting minutes
    if predictions.item() == 1:
        minutes = True
    else:
        minutes = False

    # Print the minutes
    print(minutes)


# In[ ]:


### THE END ###

