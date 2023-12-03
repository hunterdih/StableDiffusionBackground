from PIL import Image
import numpy as np

def blend_images(image1_path, image2_path, output_path, resize_width=0, resize_height=0, blend_width=30, direction='horizontal'):
    """
    Blends two images side by side (horizontal) or one on top of the other (vertical) 
    with a given width for the blending region.

    :param image1_path: The file path for the first image.
    :param image2_path: The file path for the second image.
    :param output_path: The file path to save the blended image.
    :param blend_width: The width of the region over which to blend the two images.
    :param direction: The direction to blend the images, 'horizontal' or 'vertical'.
    :return: The file path of the blended image.
    """
    # Load the images
    if resize_height !=0 and resize_width!=0:
        image1 = Image.open(image1_path).resize((resize_width // 2, resize_height // 2))
        image2 = Image.open(image2_path).resize((resize_width // 2, resize_height // 2))
    else:
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

    # Convert to numpy arrays for manipulation
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Check the direction for blending
    if direction == 'horizontal':
        # Check if the heights are the same for horizontal blending
        assert image1_array.shape[0] == image2_array.shape[0], "Images must be the same height for horizontal blending!"

        # The total width is the sum of the two images minus the blend region we will overlap
        total_width = image1_array.shape[1] + image2_array.shape[1] - blend_width
        combined_array = np.zeros((image1_array.shape[0], total_width, image1_array.shape[2]), dtype=np.uint8)
        
        # Place the first image into the combined array
        combined_array[:, :image1_array.shape[1]] = image1_array
        
        # Blend the overlapping regions horizontally
        for i in range(blend_width):
            alpha = i / blend_width
            combined_array[:, image1_array.shape[1] - blend_width + i] = (
                image2_array[:, i] * alpha + image1_array[:, image1_array.shape[1] - blend_width + i] * (1 - alpha)
            ).astype(np.uint8)
        
        # Place the second image into the combined array
        combined_array[:, image1_array.shape[1]:] = image2_array[:, blend_width:]
    
    elif direction == 'vertical':
        # Check if the widths are the same for vertical blending
        assert image1_array.shape[1] == image2_array.shape[1], "Images must be the same width for vertical blending!"

        # The total height is the sum of the two images minus the blend region we will overlap
        total_height = image1_array.shape[0] + image2_array.shape[0] - blend_width
        combined_array = np.zeros((total_height, image1_array.shape[1], image1_array.shape[2]), dtype=np.uint8)
        
        # Place the first image into the combined array
        combined_array[:image1_array.shape[0], :] = image1_array
        
        # Blend the overlapping regions vertically
        for i in range(blend_width):
            alpha = i / blend_width
            combined_array[image1_array.shape[0] - blend_width + i, :] = (
                image2_array[i, :] * alpha + image1_array[image1_array.shape[0] - blend_width + i, :] * (1 - alpha)
            ).astype(np.uint8)
        
        # Place the second image into the combined array
        combined_array[image1_array.shape[0]:, :] = image2_array[blend_width:, :]

    # Convert back to an image
    combined_image = Image.fromarray(combined_array)

    # Save the result
    combined_image.save(output_path)

    return output_path