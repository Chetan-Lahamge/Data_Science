import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(_, _), (test_images, test_labels) = mnist.load_data()

# Preprocess the data: Reshape and normalize
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

# Load the pre-trained model
model = load_model('mnist_model.keras')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# Output the evaluation results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# You can also make predictions on the test set
predictions = model.predict(test_images)

# Display a few predictions along with their test images (Optional)
for i in range(5):  # Display the first 5 predictions
    predicted_digit = np.argmax(predictions[i])  # Get the predicted label
    actual_digit = test_labels[i]  # Actual label
    
    # Plot the image
    plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {predicted_digit}, Actual: {actual_digit}")
    plt.axis('off')  # Hide axis labels
    plt.show()