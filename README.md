# kaggle-google-quest

Code for training the transformer models of the 6th place solution in the Google QUEST Q&A Labeling Kaggle competition.

A detailed description is posted on the Kaggle discussion section [here](https://www.kaggle.com/c/google-quest-challenge/discussion/130243)

Running the code requires:
- The competition data put in a directory named `data`. It can be downloaded from [https://www.kaggle.com/c/google-quest-challenge/data](https://www.kaggle.com/c/google-quest-challenge/data). 
- The packages in the requirements.txt file need to be installed. 
- By default training is done on GPU (single RTX 2080Ti) so CUDA needs to be available as well as about 10GB of GPU memory. To adress CUDA out of memory errors, the batch size can be lowered to 1 and gradient accumelation raised to 8 inside the `train.py` script.

To reproduce all 4 transformer models run the following command: 
```
python train.py -model_name=siamese_roberta && finetune.py -model_name=siamese_roberta\
train.py -model_name=siamese_bert && finetune.py -model_name=siamese_bert\
train.py -model_name=siamese_xlnet && finetune.py -model_name=siamese_xlnet\
train.py -model_name=double_albert && finetune.py -model_name=double_albert\
```

The notebooks folder contains two notebooks. The `stacking.ipynb` implements our weighted ensembling + post-processing grid search and the `oof_cvs.ipynb` shows the CV scores of our models under variuos settings (i.e. ignoring hard targets or ignoring duplicate question rows).
