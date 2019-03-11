#Step 01: Import các thư viện cần thiết
import os
import random
import numpy as np
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

#Step 02: Lấy các đường path dẫn đến tập dữ liệu Training và Testing
ROOT_PATH = "./traffic"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")
#print(os.path.abspath(train_data_dir))
#----------------------------------------------
#Step 03: Đọc/Load dữ liệu ra các mảng
# Function Load a data set and returns two lists images and corressponding labels
def load_data(data_dir):
    """ 
        Loads a data set and returns two lists:
           - images: a list of Numpy arrays, each representing an image.
           - labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # Declare Two lists: labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

#----------------------------------------------
#Step 04: HANDLING IMAGES OF DIFFERENT SIZES ==> RESIZE FOR IMAGES
print("Training Dataset is loading......")
images, labels = load_data(train_data_dir)
print("Training Dataset is loaded!")
print("-------------------------------------------------------")
print("Before resize images to 32x32 [images]")
for image in images[:5]:
   print("shape: {0}, \
    min: {1}, max: {2}"\
        .format(image.shape, image.min(), image.max()))
print("-------------------------------------------------------")
# Resize images to 32x32
images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
               for image in images]

def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()
#display_images_and_labels(images32, labels)
#
print("-------------------------------------------------------")
print("After resize images to 32x32 [images32]")
for image in images32[:5]:
    print("shape: {0}, \
    min: {1}, max: {2}"\
        .format(image.shape, image.min(), image.max()))
print("-------------------------------------------------------")

#plt.subplot(211)
#plt.imshow(images[0])
#plt.subplot(212)
#plt.imshow(images32[0])
#plt.show()

labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)
#-------------Minimum Viable Model--------------------------

# Create a graph to hold the model.
graph = tf.Graph()
# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To Input Layer: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # 1 Hidden layer: 1024 unit
    xx1 = tf.contrib.layers.fully_connected(images_flat, 128, tf.nn.relu)
    xx2 = tf.contrib.layers.fully_connected(xx1, 128, tf.nn.relu)
    # Output layer
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(xx2, 62, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()

print("images_flat: ", images_flat)
#print("hidden layer: ",xx)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)
print("Bat dau train")
#-------------TRAINING--------------------------
# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init])
for i in range(1001):
    _, loss_value = session.run([train, loss], 
                                feed_dict=
                                {images_ph: images_a, 
                                labels_ph: labels_a})
    if i % 100 == 0:
        print("Loss: ", loss_value)

#-------------USING THE MODEL--------------------------
# Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, 
             "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])
    
plt.show()

#-------------EVALUATION--------------------------
# Load the test dataset.
test_images, test_labels = load_data(test_data_dir)
# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                 for image in test_images]
#display_images_and_labels(test_images32, test_labels)
# Run predictions against the full test set.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))
# Close the session. This will destroy the trained model.
session.close()