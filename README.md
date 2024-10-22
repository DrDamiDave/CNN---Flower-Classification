# CNN-Flower-Classification
This project demonstrates a machine learning workflow aimed at classifying images of flowers into five distinct categories using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras. The dataset used is a collection of flower images, which is publicly available through TensorFlow's dataset repository.

#### 1. **Data Acquisition and Setup**
The flower dataset consists of images divided into five classes: **roses**, **daisies**, **dandelions**, **sunflowers**, and **tulips**. The dataset is downloaded and extracted, with each image organized in folders according to its category. These images are later read into the program, and each category is assigned a numerical label for use in the machine learning process.

#### 2. **Data Preprocessing**
Before feeding the images into the neural network, the data undergoes preprocessing to ensure it is uniform and ready for training. This involves:
- **Reading the Images**: The images are loaded into memory using OpenCV.
- **Resizing**: All images are resized to a fixed size of 180x180 pixels, ensuring consistent input size for the neural network.
- **Normalization**: The pixel values of the images, which originally range from 0 to 255, are normalized to a range of 0 to 1. This is done by dividing each pixel value by 255, improving model convergence during training.
- **Splitting the Dataset**: The data is divided into training and testing sets, allowing the model to learn on the training set and evaluate its performance on the unseen test set.

#### 3. **Model Architecture**
The Convolutional Neural Network (CNN) model used in this project is a **sequential model** that includes multiple layers designed for feature extraction and classification:
- **Convolutional Layers**: The network starts with three convolutional layers, each progressively learning more complex features of the images. These layers use filters to scan across the image and detect patterns such as edges, textures, and shapes.
- **Pooling Layers**: After each convolutional layer, max-pooling layers are employed to reduce the spatial dimensions of the feature maps. Pooling helps in reducing the number of parameters and prevents overfitting by summarizing feature information.
- **Fully Connected Layers**: After the feature extraction process, the output from the convolutional layers is flattened and passed through dense layers. These layers process the high-level features to classify the images into one of the five flower categories.
- **Activation Functions**: The ReLU activation function is applied to introduce non-linearity, while the final output layer uses a softmax function to predict the probability distribution over the five classes.

#### 4. **Model Compilation and Training**
The model is compiled with the **Adam optimizer**, which adapts the learning rate during training for faster and more efficient convergence. The **Sparse Categorical Crossentropy** loss function is used, which is ideal for multi-class classification problems where the labels are integers. The network is trained on the scaled image data for multiple epochs, adjusting its weights iteratively to minimize the loss and improve accuracy.

#### 5. **Data Augmentation**
To further enhance the performance of the model and prevent overfitting, **data augmentation** is applied. This technique artificially expands the dataset by creating variations of the existing images through transformations such as random flips, rotations, and zooms. This helps the model generalize better when exposed to new, unseen data.

#### 6. **Evaluation**
After training, the model is evaluated on the test set to assess its performance. The evaluation provides metrics such as accuracy, which indicates how well the model is able to correctly classify flower images it hasn't seen before. The predictions are also generated to see the model's confidence in categorizing each flower image.

#### 7. **Conclusion**
This project demonstrates how convolutional neural networks can effectively classify images into distinct categories, leveraging data preprocessing, a well-structured CNN architecture, and techniques like data augmentation. The combination of these elements results in a robust model capable of achieving high accuracy in flower classification tasks.
