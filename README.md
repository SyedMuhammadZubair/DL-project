# DL-project
# BIGGEST CHALLENGE: 
Most challenging part of implementation is the loss function we get resolved by using the triplet loss 
as we have discussed in the objective function section. Another challenging issue we face to arrange 
the dataset and make triplets, so we discuss it in custom dataset class.
# CUSTOM DATASET CLASS:
We have implemented our own Facial Recognition model using pytorch and LFW dataset. To load the 
dataset from the disk we use two functions i.e., get_item and get_length. Mainly focused on dataset 
and objective function component, we are using LFW dataset’s images, by using OS module to fetch 
images from the folders that containing images more than one, one image for anchor class and one 
image for positive class, folders have one image are used for negative class. We have created a .csv file 
to store paths of anchor, positive, and negative class images, and the positive class for labels. Some 
image folders contain only a single image so we consider it as a negative image class.
# REMOVE TRIPLETS: 
From anchor, positive and negative we remove the soft/easy triplets by using clamp function with 
margin fixed on minimum 0.0 to find the distance between positive and negative class images from 
anchor. 
# OBJECTIVE FUNCTION:
Triplet loss function is defined to calculate the loss, function gets margin as input in our case we use 
margin of two, function use the index, anchor, positive, and negative images than calculate the positive 
distance from anchor and positive image also negative distance from anchor and negative image, after 
that both distances pass to the clamp function with minimum margin of zero for each index. Function 
returns the mean of all individually calculated losses. We run 5 epochs with the batch size 4.
# MODEL:
To train our model we use Siamese Network to train the prediction model. The model consists of two 
conv layers with ReLU as activation function, with batch normalization and dropout layers after max 
pooling is used.
