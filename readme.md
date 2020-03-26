1. Data preparation: 
- Data is taken from camera's laptop at various conditions (light, distance, background). Included 24 static sign language characters.
- Apply some simple augmentation technique: rotate, scale, ... (by keras library)
- Preprocessing: Background subtraction (using skin color range in HSV space).
2. Buiding the model:
- Build a small neural network to predict these sign language. Increasing the complexity of model until validation loss increase (and training loss is still decrease). Apply some regularization techniques: early stopping, step decay for learning rate.
- Use Google Colabotary for training period:

![training accuracy](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/training_accuracy_graph%20(3).png)

![training loss](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/training_loss_entropy_graph%20(3).png)

![validation accracy](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/validation_accuracy_graph%20(3).png)

![validation loss](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/validation_loss_entropy_graph%20(3).png)

3. Results:
- Model needs good light condition to perform well. Sensitive to light and noise.
- Some character have bad results: X, M, N because after preprocessing stage, they look similar.
- Details is in Report.docm file.



