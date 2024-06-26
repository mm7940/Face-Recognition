
# Face Recognition

**architecture.py**  

This file generate our architecture for the CNN dynamically just by providing args in the constructor.
The classes Layer1, Layer2 and Layer3 are constant which are used in Architecture class to generate the model as follows:  
Layer1 -- args['layer1'] times  
Layer2 -- args['layer2'] times  
Flatten  
Layer3 -- args['layer3'] times  
Linear  
Softmax

**utils.py**  

This file has 2 functions: 

plot: Used to plot the graph of train loss and validation loss in same subgraph and train accuracy and validation accuracy in another same subplot. This also saves the plot generated.  

train: The arguments it takes are the model, train dataset, validation dataset and some other hyperparameters to train on the train dataset. It saves the final model and returns the train and validation losses and train and validation accuracies. 

**train.ipynb**  

This file has 3 jobs:  
    1. Create the dataset for training and validation.  
    2. Create the architecture using architecture.py file.  
    3. Train the parameters for architecture using train function from utils.py file.  

All the three jobs have different set of hyperparameters for preprocessing, model generation and training respectively which needs to be tuned properly for maximum validation accuracy.

This file was run on GPU in google colab to generate the model.pt and names.pkl file inside the modelname/ folder for differtent models.  

**predict.ipynb**  

This is the test file of our created model. It picks the model from ./modelname/model.pt and names from ./modelname/names.pkl.  
The images for which we want to predict the name has to be provided as the path of the image in predict_images array. The person must be present in the names array for which we want to predict the label.  
Finally it predicts the name and gives output as true name, predicted name and image of the person.  

**resnet152.ipynb**

This file loads the resnet model and modifies its last layer. Then after training it saves the model. If you wish to train the model from scratch just follow the cells. If you wish to train on our pretrained model then run the cell which loads the model and optimizer parameters before running the training loop.


**Directory**

 ---
Project  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- Dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- train.ipynb  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- predict.ipynb  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- utils.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- architecture.py  

Changing the modelname and hyperparameters in train.py and running it will generate a folder in Project directory with name as modelname and has the following contents:

modelname  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- model.pt  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- name.pkl  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- plot.png  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- graph.png  

Changing the modelname in the predict.ipnb file will take the model.pt and names.pkl file from that model folder to predict. We can generate different models and compare the predictions by them.  