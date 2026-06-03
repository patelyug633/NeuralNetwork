#
# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt
from os.path import join  # Add this import

#
# Set file paths based on added MNIST Datasets
#
input_path = 'C:\\Users\\patel\\Desktop\\Projects\\NeuralNetwork\\MnistData'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Define the MnistDataloader class (since it's missing)
#
class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath, 
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def load_data(self):
        # Load training data
        x_train = self.load_images(self.training_images_filepath)
        y_train = self.load_labels(self.training_labels_filepath)
        
        # Load test data
        x_test = self.load_images(self.test_images_filepath)
        y_test = self.load_labels(self.test_labels_filepath)
        
        return (x_train, y_train), (x_test, y_test)
    
    def load_images(self, filepath):
        """Load IDX/UByte format image files"""
        with open(filepath, 'rb') as f:
            # Read magic number (first 4 bytes)
            magic = int.from_bytes(f.read(4), 'big')
            
            # Number of images (next 4 bytes)
            num_images = int.from_bytes(f.read(4), 'big')
            
            # Rows and columns (next 4 bytes each)
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # Read the actual image data
            image_data = f.read()
            
            # Convert to numpy array
            import numpy as np
            images = np.frombuffer(image_data, dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
            
            return images
    
    def load_labels(self, filepath):
        """Load IDX/UByte format label files"""
        with open(filepath, 'rb') as f:
            # Read magic number (first 4 bytes)
            magic = int.from_bytes(f.read(4), 'big')
            
            # Number of labels (next 4 bytes)
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # Read the actual label data
            label_data = f.read()
            
            # Convert to numpy array
            import numpy as np
            labels = np.frombuffer(label_data, dtype=np.uint8)
            
            return labels

#
# Load MNIST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                   test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 1):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 1):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)
plt.show() 