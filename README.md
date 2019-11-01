# Emotion_Recognition_KERC-kaggle-
Emotion Recognition based fine-tuning(various Model)


# dependency

  $ pip install -r requirements.txt

# pre_train

  1. Run **pretrain.py** file. (You need to put the data in the pretrainingData folder in csv file format, **like FER2013** / And you choose pretrain model int OtherModel)
    and then, The weights file is created in the pre_trained folder
  
  2. run **get_mainface_frame.py** file to extract the main face in each frame of the video clip. You need to put the right path of the data folder.
  
  3. run **data_rearrange.py** file to create the "data_file.csv" to save the information of data. You need to put the right path of the data folder of Step 1 result. For detail, we copy the sub folder train and val from folder "data_try_out" to folder "data" and then run file data_rearrange.py.
  
  4. run **extract_features.py** file to extract the feature of the data by VGG16 pretrain on VGGFace dataset. (you can change the other pretrain model in the extractor.py file). Running this file will create sub folder "sequences" in folder "data".
  
  5. run **train.py** file to training the baseline model (you can create new model in the models.py file).

