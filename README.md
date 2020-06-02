# Project Name : Emotion-recognition
# Table of Content :
1.[Description](#p1)

2.[Installations](#p2)

3.[Usage](#p3)

4.[Dataset](#p4)






<a id="p1"></a> 
# Description:

This is the fork of https://github.com/omar178/Emotion-recognition with more emphasize to enlarge the variants of different models + GPU support and ploting options
## What does Emotion Recognition mean?

Emotion recognition is a technique that allows a program to "read" the emotions on a human face using advanced image processing. Companies have been experimenting with combining sophisticated algorithms with image processing techniques that have emerged in the past ten years to understand more about what an image or a video of a person's face tells us about how he/she is feeling and not just that but also showing the probabilities of mixed emotions a face could has.

<a id="p2"></a> 
# Installations:

Install dependencies using requirements.txt

```shell
pip install -r requirements.txt
```

<a id="p3"></a> 
# Usage:

The program will creat a window to display the scene capture by webcamera and a window representing the probabilities of detected emotions.

> Demo

python real_time_video.py

You can just use this with the provided pretrained model i have included in the path written in the code file, i have choosen this specificaly since it scores the best accuracy, feel free to choose any but in this case you have to run the later file train_emotion_classifier
> If you just want to run this demo, the following content can be skipped
- Train

- python train_emotion_classifier.py


<a id="p4"></a> 
# Dataset:

I have used [this](https://www.kaggle.com/c/3364/download-all) dataset

Download it and put the csv in fer2013/fer2013/

-fer2013 emotion classification test accuracy: 65%


# Credits
This work is inspired from [this](https://github.com/oarriaga/face_classification) great work.
