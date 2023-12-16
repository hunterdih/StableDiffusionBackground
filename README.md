
# StableDiffusionBackground Generator

Welcome to the StableDiffusionBackground Generator! This tool allows you to generate detailed images based on textual prompts using the power of machine learning and the Stable Diffusion model.

## Getting Started

### Prerequisites
- Git (for cloning the repository)
- Anaconda (for creating and managing the environment)
- Python 3.8 or later
- Packages: Hugging Face Diffuser, NumPy, PyTorch (CUDA)

### Installation

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/hunterdih/StableDiffusionBackground.git
   ```
2. Navigate into the cloned repository directory:
   ```
   cd StableDiffusionBackground
   ```
3. Create a new Anaconda environment:
   ```
   conda create --name stable_diffusion python=3.8
   ```
4. Activate the Anaconda environment:
   ```
   conda activate stable_diffusion
   ```
5. Install the required packages

## Usage

After installation, you can run the program from the command line. The script supports various options that you can specify to customize the output.

### Command-Line Options

Here's a list of available options you can use:

- `--prompt`: The image prompt. (required)
- `--width`: Initial width of the image. (default: 640)
- `--height`: Initial height of the image. (default: 360)
- `--desired_width`: Desired width after increasing resolution. (default: 2560)
- `--desired_height`: Desired height after increasing resolution. (default: 1440)
- `--overlap`: Overlap size for image parts. (default: 30)
- `--blend_width`: Width for blending image parts. (default: 120)
- `--output`: Optional output directory suffix. If set, changes the output directory.
- `--increase_res`: Flag to increase resolution. (default: True)

To generate an image, use the following command:

```
python stable_diffusion_background.py --prompt "astronaut frog in space"
```

If you want to customize the dimensions and other settings:

```
python stable_diffusion_background.py --prompt "a puffer fish with a tiara" --width 800 --height 450 --desired_width 3200 --desired_height 1800
```

The program will produce a low-resolution preview of the image. If you're satisfied with the preview, the high-resolution image generation will proceed.

### Saving Images

By default, the program saves all images and intermediate files in the `outputs/` directory followed by the prompt. If you specify the `--output` option, it will save them in `outputs/[your_output_suffix]`.

### Documentation

To view the complete documentation and all available options, you can invoke the help command:

```
python stable_diffusion_background.py --help
```

This will display all the command-line options along with their descriptions.

Enjoy creating with the StableDiffusionBackground Generator!
