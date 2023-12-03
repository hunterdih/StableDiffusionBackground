from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt

# Load the images
image1_path = 'test/houses.jpg'
image2_path = 'test/stairs.jpg'
image1 = Image.open(image1_path).convert("RGBA")
image2 = Image.open(image2_path).convert("RGBA")

desired_width = 900
desired_height = 900

# Ensure the images are the same size
image1 = image1.resize((desired_width, desired_height))
image2 = image2.resize((desired_width, desired_height))

# Create an initial mask that defines the feature for distance transform
width, height = image1.size
initial_mask = np.zeros((height, width))
# For example, draw a line in the middle of the mask
initial_mask[:, width // 2:] = 1  # Change this line to modify the feature for the distance transform

# Apply the distance transform to the mask
distance_mask = distance_transform_edt(1 - initial_mask)

# Normalize the distance mask to range [0, 1]
distance_mask = distance_mask / np.max(distance_mask)

# Create the alpha mask from the distance mask
alpha_mask = Image.fromarray((distance_mask * 255).astype("uint8"))

# Apply the mask to the second image
image2.putalpha(alpha_mask)

# Alpha composite the images together
combined_image = Image.alpha_composite(image1, image2)

# Save the blended image
combined_image.save('test/blended_image.png')
