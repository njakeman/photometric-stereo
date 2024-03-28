# Uncalibrated Photometric Stereo

This script was generated through iterations with Chat-GPT 4 (27/03/2024).

The conda environment used sucessfully locally is at Python 3.10.14 with requirements found in the associated txt file.

Simple use:

`python photometric_stereo.py <path/to/image_folder> [path/to/mask_file.jpg/jpeg/png]`

Currrently this struggles to process large, images natively taken on a DSLR.
I used imagemagick to resize images to 25% and convert to grayscale.

`magick convert <input> -set colorspace Gray -separate -average <output>`

`magick convert <input> -resize 25% <output>`

I've tried this with the *most* rudimentary setup and achieved *plausible* results with the files supplied. The script outputs OpenGL (Blender compatible) and DirectX format normal maps into the folder from which it runs.







