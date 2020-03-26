- Data is taken from camera's laptop at various conditions (light, distance, background). Included 24 static sign language characters.
- Preprocessing: Background subtraction (using skin color range in HSV space).
- Build a small neural network to predict these sign language.
- Use Google Colabotary for training period:

![training accuracy](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/training_accuracy_graph%20(3).png)

![training loss](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/training_loss_entropy_graph%20(3).png)

![validation accracy](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/validation_accuracy_graph%20(3).png)

![validation loss](https://github.com/hhd-bk/Vietnamese-Sign-Language-Recognition/blob/master/Graph/validation_loss_entropy_graph%20(3).png)

- Model needs good light condition to perform well.
- Some character have bad results: X, M, N because after preprocessing stage, they look similar.
- Details is in Report.docm file.



