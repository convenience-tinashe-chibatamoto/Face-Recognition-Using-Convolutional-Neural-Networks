  <h1>Face Recognition</h1>
  <p>This is a Python-based face recognition system that can detect and recognise my face in group images.</p>

  <h2>Features</h2>
    <li><strong>Face Recognition</strong>: The program is designed to perform face recognition, where it can detect and recognize faces in images and video streams.</li>
    <li><strong>Custom Dataset Training</strong>: The program supports training the machine learning model on a custom dataset of images. The <code>make_train_data</code> function is used to create the dataset for training.</li>
    <li><strong>Convolutional Neural Network (CNN)</strong>: The program uses a Convolutional Neural Network (CNN) architecture for the face recognition task. The CNN model is defined and trained in the code.</li>
    <li><strong>Image Preprocessing</strong>: The program preprocesses the input images by resizing them to a fixed size (300x300 pixels) and normalising the pixel values.</li>
    <li><strong>Data Augmentation</strong>: The program uses the <code>ImageDataGenerator</code> from TensorFlow to perform data augmentation on the training data, which helps improve the model's generalisation.</li>
    <li><strong>Visualization</strong>: The program includes code to visualise the training and validation loss, as well as the training and validation accuracy, during the model training process.</li>
    <li><strong>Prediction</strong>: The program includes a function <code>predict_with_class</code> that can be used to make predictions on new images and display the classification results (whether the person is "In Picture" or "Not In Picture").</li>
    <li>My aim with this program was to demonstrate a complete end-to-end face recognition system, including data preparation, model training, and prediction on new images. The use of a CNN architecture and data augmentation techniques are common approaches for building robust face recognition models.</li>

  <h2>Requirements</h2>
  <ul>
    <li>Python 3.x</li>
    <li>OpenCV: For image processing and manipulation.</li>
    <li>TensorFlow: For deep learning and neural network operations.</li>
    <li>Numpy: For numerical operations and array manipulation.</li>
    <li>Matplotlib: For data visualisation.</li>
    <li>Pandas: For data manipulation and analysis.
    <li>scikit-learn: For train-test split and label encoding.</li>
    <li>tqdm: For progress bar display during long-running operations.</li>
  </ul>

  <h2>Usage</h2>
  <ol>
    <li>Clone the repository:
      <pre><code>git clone https://github.com/convenience-tinashe-chibatamoto/Face-Recognition.git</code></pre>
    </li>
    <li>Install the required dependencies:
      <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Train the model on your custom dataset.
    </li>
  </ol>

  <h2>Contributing</h2>
  <p>Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.</p>

  <h2>License</h2>
  <p>This project is licensed under the <a href="LICENSE">MIT License</a>.</p>
