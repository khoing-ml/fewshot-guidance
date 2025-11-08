import numpy as np
from PIL import Image
import os

def npy_to_image(input_npy_path, output_image_path):
    """
    Converts a NumPy array stored in a .npy file to a standard image file.

    :param input_npy_path: Path to the input .npy file.
    :param output_image_path: Path to save the output image (e.g., 'output.png').
    """
    try:
        # 1. Load the NumPy array from the .npy file
        print(f"Loading array from: {input_npy_path}...")
        numpy_array = np.load(input_npy_path)
        print(f"Array loaded. Shape: {numpy_array.shape}, Data Type: {numpy_array.dtype}")

        # --- CRITICAL PRE-PROCESSING ---
        
        # Ensure the array is 2D (Grayscale) or 3D (RGB)
        # Squeeze to remove single-dimensional entries (e.g., a shape of (1, 256, 256) becomes (256, 256))
        if len(numpy_array.shape) > 2 and numpy_array.shape[0] == 1:
            numpy_array = np.squeeze(numpy_array, axis=0)
        
        # 2. Rescale and Convert Data Type to uint8 (0-255)
        # Image libraries typically expect pixel values in the 0-255 range and np.uint8 format.
        if numpy_array.dtype != np.uint8:
            print("Converting data type and rescaling...")
            
            # If data is float (e.g., 0.0 to 1.0), rescale to 0-255
            if np.issubdtype(numpy_array.dtype, np.floating):
                # Ensure values are clipped to avoid overflow if array contains values slightly > 1.0
                numpy_array = np.clip(numpy_array, 0.0, 1.0)
                numpy_array = (numpy_array * 255).astype(np.uint8)
            
            # If data is large integer (e.g., 16-bit, 32-bit), you'll need a custom scaling logic 
            # based on your array's value range (e.g., dividing by max value and multiplying by 255).
            elif np.issubdtype(numpy_array.dtype, np.integer):
                 # Simple cast for typical 0-255 integer data, otherwise, proper scaling is needed
                 numpy_array = numpy_array.astype(np.uint8)

        # 3. Convert array to PIL Image object
        if len(numpy_array.shape) == 2:
            # Grayscale image (Height, Width)
            mode = 'L' 
        elif len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
            # RGB color image (Height, Width, 3)
            mode = 'RGB'
        elif len(numpy_array.shape) == 3 and numpy_array.shape[2] == 4:
            # RGBA color image (Height, Width, 4)
            mode = 'RGBA'
        else:
            raise ValueError(f"Unsupported array shape for image conversion: {numpy_array.shape}. Must be (H, W) or (H, W, 3/4).")

        img = Image.fromarray(numpy_array, mode=mode)

        # 4. Save the image
        img.save(output_image_path)
        print(f"Success! Image saved to: {output_image_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_npy_path}")
    except ValueError as e:
        print(f"Error during conversion: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- EXAMPLE USAGE ---

# 1. Define File Paths
NPY_FILE = 'unconditional_sample.npy'
OUTPUT_FILE = 'converted_image.png'

# 2. (OPTIONAL) Create a sample .npy file if you don't have one
if not os.path.exists(NPY_FILE):
    print(f"Creating a sample RGB array: {NPY_FILE}")
    # Create a 100x100 RGB array with random data (0 to 255)
    sample_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Make the top-left quadrant red for easy verification
    sample_array[:50, :50] = [255, 0, 0] 
    np.save(NPY_FILE, sample_array)

# 3. Run the conversion
npy_to_image(NPY_FILE, OUTPUT_FILE)

# (Optional: Display the converted image)
# Image.open(OUTPUT_FILE).show()