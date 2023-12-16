To use the StableDiffusionBackground generator, first clone the respository and create an anaconda environment.
- To generate an image, you'll need to install several essential packages: the Huggingface Diffusers package, NumPy, and PyTorch with CUDA support.
- First, assign your desired image description to the variable named 'prompt'.
- When you execute the program, it initially produces a low-resolution preview of the image. You can then decide if you want to enhance the resolution.
- Should you choose to increase the image resolution, all relevant files and sub-files will be saved in a folder named 'outputs' followed by the specific prompt you used.
