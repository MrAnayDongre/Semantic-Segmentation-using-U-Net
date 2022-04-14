


## This project illustrates how to implement sematic segmentation with the help of U-Net architecture

## 1. [Basic U-Net Architecture](https://arxiv.org/abs/1505.04597)


![u-net-architecture](https://user-images.githubusercontent.com/56868253/163403861-eec8fbb0-12b1-4be9-968f-fb8c8733ef2a.png) 


## 2. [Kaggle Dataset which are goingo to train our model on](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)


Satellite images of Dubai, the UAE segmented into 6 classes
The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles. The classes are:

1. Building: #3C1098

2. Land (unpaved area): #8429F6

3. Road: #6EC1E4

4. Vegetation: #FEDD3A

5. Water: #E2A929

6. Unlabeled: #9B9B9B

![Sematic Seg Kaggle](https://user-images.githubusercontent.com/56868253/163405694-298cfcaa-6e57-43ab-a7bb-bfadf7a620e0.png)


## 4. After pathifying and One-hot encoding :
![After one-hot encoding](https://user-images.githubusercontent.com/56868253/163407017-d4f77bcf-cf41-4393-a30d-1850bea4e2a1.png)


## 5. Model summary which we are going to implement using TensorFlow.
![Model_summary](https://user-images.githubusercontent.com/56868253/163407138-2e49dea8-5415-4359-8ba2-099314a9a179.png)

## 6. Training for 30 epochs and bathx_size of 16:
![Training images](https://user-images.githubusercontent.com/56868253/163407631-fd322d41-745c-4f6b-94f6-7c56f61d3234.png)

## 7. Training and Validation loss:
![Training and val loss](https://user-images.githubusercontent.com/56868253/163407996-c9d99e35-f7ae-4381-9ae5-eba4bf51d6b9.png)

## 8. Training and validation IoU:
![Training and val IoU](https://user-images.githubusercontent.com/56868253/163408087-f71e298b-6af0-4eeb-a75d-e9f518dd3aa9.png)


## 9. MODEL PREDICTION AFTER TRAINING USING U-Net Architecture
![Model Predcition](https://user-images.githubusercontent.com/56868253/163408321-a3492497-b4dd-4d59-b4db-a1100898ac62.png)


### THANK YOU FOR YOU TIME !
