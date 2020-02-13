# kaggle-google-quest

Code for training the transformer models of the 6th place solution in the Google QUEST Q&A Labeling Kaggle competition.

To reproduce a trained model (e.g. Siamese Roberta) run: 
```
python train.py -model_name=siamese_roberta && finetune.py -model_name=siamese_roberta
```