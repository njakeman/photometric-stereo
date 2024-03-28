import numpy as np
import cv2
import os
from scipy.linalg import lstsq
import sys

def estimate_light_directions(images):
    light_dirs = []
    for img in images:
        y, x = np.unravel_index(np.argmax(img), img.shape)
        z = np.sqrt(max(1 - x**2 - y**2, 0))
        light_dirs.append([x, y, z])
    return np.array(light_dirs)

def convert_to_normal_map(normals, format='DirectX'):
    normal_map = np.zeros_like(normals)
    for i in range(normals.shape[0]):
        for j in range(normals.shape[1]):
            n = normals[i, j]
            if format == 'OpenGL':
                n[1] = -n[1]
            normal_map[i, j] = n * 0.5 + 0.5
    return normal_map
    

def main(image_dir, mask_path=None):
    # Load images from the directory
    images = [cv2.imread(os.path.join(image_dir, img), cv2.IMREAD_GRAYSCALE) 
              for img in sorted(os.listdir(image_dir)) 
              if img.endswith(('.png', '.jpg', '.jpeg'))]
              
    # Initialize the mask
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Mask image {mask_path} not found.")
            sys.exit(1)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = np.ones(images[0].shape, dtype=np.uint8) * 255

    # Estimate light directions
    light_dirs = estimate_light_directions(images)

    # Compute normals and albedo
    height, width = images[0].shape
    normals = np.zeros((height, width, 3))
    albedo = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if mask[i, j] == 255:  # Only consider pixels within the mask
                I = np.array([img[i, j] for img in images])
                N, _, _, _ = lstsq(light_dirs, I)
                albedo[i, j] = np.linalg.norm(N)
                normals[i, j] = N / (np.linalg.norm(N) + 1e-5)
            else:
                normals[i, j] = [0, 0, 1]  # Default normal for masked-out areas

    # Generate normal maps
    directx_normal_map = convert_to_normal_map(normals, 'DirectX')
    opengl_normal_map = convert_to_normal_map(normals, 'OpenGL')

    # Save normal maps
    cv2.imwrite('directx_normal_map.png', directx_normal_map * 255)
    cv2.imwrite('opengl_normal_map.png', opengl_normal_map * 255)

#if __name__ == "__main__":
#    if len(sys.argv) != 2:
#        print("Usage: python photometric_stereo.py <image_directory>")
#        sys.exit(1)
#    main(sys.argv[1])

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python photometric_stereo.py <image_directory> [<mask_image_path>]")
        sys.exit(1)

    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1])