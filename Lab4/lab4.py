import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import inf
def read_pfm(file_path):

    with open(file_path, "rb") as f:
        # Read the header
        header = f.readline().decode().rstrip()
        if header not in ["PF", "Pf"]:
            raise ValueError("Not a PFM file.")

        color = header == "PF"

        dims = f.readline().decode().strip()
        width, height = map(int, dims.split())

        scale = float(f.readline().decode().strip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        data = np.reshape(data, (height, width, 3) if color else (height, width))
        data = np.flipud(data)  # Flip vertically due to PFM format
        data = np.array(data)
        data[np.isinf(data)] = 0
        return data, scale

def display_disparity_map(img, plt_label = 'Map'):

    # Display the disparity map
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='grey')
    plt.title(plt_label)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def disparity_to_depth(disparity_map, baseline, focal_length, aspect =1):
    valid_disparity = disparity_map > 0  
    adjusted_disparity = disparity_map
    depth_map = np.zeros_like(disparity_map)
    depth_map[valid_disparity] = baseline * focal_length / adjusted_disparity[valid_disparity]

    depth_map = depth_map / aspect
    return depth_map

def compute_disparity_map_SGBM(left_image_path, right_image_path, min_disparity=0, num_disparities=16, block_size=5):
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        raise ValueError("One or both input images could not be loaded.")
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("Block size must be an odd number and >= 3.")
    if num_disparities % 16 != 0:
        raise ValueError("Number of disparities must be divisible by 16.")

    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2, disp12MaxDiff=2,
        uniquenessRatio=15, speckleWindowSize=100, speckleRange=1, preFilterCap=63, )

    print("Computing disparity map using StereoSGBM...")
    disparity_map = stereo_sgbm.compute(left_img, right_img).astype(np.float32) /16.0

    return disparity_map

def depth_map_normalize_8bit(depth_map):
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

    return depth_map_normalized

def depth_map_normalize_24bit(depth_map):

    normalized_depth = (cv2.normalize(depth_map, None, 0,  (256**3 - 1), cv2.NORM_MINMAX)).astype(np.uint32)
    # Extract RGB channels
    R = (normalized_depth & 0xFF).astype(np.uint8)
    G = ((normalized_depth >> 8) & 0xFF).astype(np.uint8)
    B = ((normalized_depth >> 16) & 0xFF).astype(np.uint8)

    # Combine channels into an RGB image
    rgb_image = np.stack((R, G, B), axis=-1)

    # Save the RGB image
    return rgb_image

def decode_uint24_depth_map(depth_map_uint24, max_depth):

    depth_map_uint24 = depth_map_uint24[:, :, :3]
    depth_map_uint24 = depth_map_uint24[:, :, ::-1]
    # Decode 24-bit depth map
    R = (depth_map_uint24[:, :, 0]).astype(np.uint32)
    G = (depth_map_uint24[:, :, 1]).astype(np.uint32) * 256
    B = (depth_map_uint24[:, :, 2]).astype(np.uint32) * 256 ** 2

    depth_map = ((R + G + B) / (2 ** 24 - 1)) * max_depth
    return depth_map

def depth_to_disparity(depth_map, baseline, focal_length):

    min_depth = 0.001
    depth_map = np.maximum(depth_map, min_depth)

    disparity_map = (baseline * focal_length) / depth_map
    disparity_map_normalized = disparity_map_normalize_8bit(disparity_map)

    return disparity_map_normalized

def disparity_map_normalize_8bit(disparity_map):
    disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_map_normalized = disparity_map_normalized.astype(np.uint8)

    return disparity_map_normalized

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

# Main function
if __name__ == "__main__":

    # Load calib data
    calib_file = r"C:\#Projects\ZAOWIR\Lab4\calib.txt"
    with open(calib_file, "r") as f:
        lines = f.readlines()
        cam0 = lines[0].strip().split("=")[1].strip("[]")
        cam1 = lines[1].strip().split("=")[1].strip("[]")
        doffs = float(lines[2].split("=")[1].strip())
        baseline = float(lines[3].split("=")[1].strip())
        # Extract focal length from cam0
        focal_length = float(cam0.split()[0])


    # Zad 1
    disparity_file = r"C:\#Projects\ZAOWIR\Lab4\disp0.pfm"
    output_path_Zad1_disparity = r"C:\#Projects\ZAOWIR\Lab4\zad1_disparity.png"
    output_path_Zad1_depth = r"C:\#Projects\ZAOWIR\Lab4\zad1_depth.png"

    disparity_map, scale = read_pfm(disparity_file)
    #display_disparity_map(disparity_map,'Disparity Map from PFM')

    # 8bit given diparity to depth
    depth_map = disparity_to_depth(disparity_map, baseline, focal_length, aspect=1000)  # zwraca w metrach
    depth_map_8bit = depth_map_normalize_8bit(depth_map)

    cv2.imwrite(output_path_Zad1_disparity, disparity_map)
    cv2.imwrite(output_path_Zad1_depth, depth_map_8bit)

    # Zad 2
    left_image_path = r"C:\#Projects\ZAOWIR\Lab4\im0.png"
    right_image_path = r"C:\#Projects\ZAOWIR\Lab4\im1.png"
    output_path_Zad2_disparity = r"C:\#Projects\ZAOWIR\Lab4\zad2_disparity.png"
    output_path_Zad2_depth = r"C:\#Projects\ZAOWIR\Lab4\zad2_depth.png"

    # Calculate my disparity
    disparity_map_sgbm = compute_disparity_map_SGBM(left_image_path, right_image_path, min_disparity=0, num_disparities=256, block_size=9)
    #display_disparity_map(disparity_map_sgbm,'Disparity Map from SGBM')

    # 8bit my disparity to depth
    depth_map_sgbm = disparity_to_depth(disparity_map_sgbm, baseline, focal_length, aspect=1000)
    depth_map_8bit_sgbm = depth_map_normalize_8bit(depth_map_sgbm)

    cv2.imwrite(output_path_Zad2_disparity, disparity_map_sgbm)
    cv2.imwrite(output_path_Zad2_depth, depth_map_8bit_sgbm)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(disparity_map, cmap='gray')
    plt.title('Disparity Map from PFM')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(disparity_map_sgbm, cmap='gray')
    plt.title('Disparity Map from SGBM')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_map_8bit, cmap='gray')
    plt.title('Depth Map from PFM')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map_8bit_sgbm, cmap='gray')
    plt.title('Depth Map from SGBM')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



    # Zad 3
    output_path_Zad3_depth_ref = r"C:\#Projects\ZAOWIR\Lab4\zad3_depth_ref.png"
    output_path_Zad3_depth_sbgm  = r"C:\#Projects\ZAOWIR\Lab4\zad3_depth_sbgm.png"

    depth_map_24bit = depth_map_normalize_24bit(depth_map)
    depth_map_24bit_sgbm = depth_map_normalize_24bit(depth_map_sgbm)

    cv2.imwrite(output_path_Zad3_depth_ref, np.array(cv2.cvtColor(depth_map_24bit, cv2.COLOR_BGR2RGB)))
    cv2.imwrite(output_path_Zad3_depth_sbgm, np.array(cv2.cvtColor(depth_map_24bit_sgbm, cv2.COLOR_BGR2RGB)))


    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_map_24bit)
    plt.title('Depth Map from PFM 24-bit')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map_24bit_sgbm)
    plt.title('Depth Map from SGBM 24-bit')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Zad 4
    output_path_Zad4_disparity = r"C:\#Projects\ZAOWIR\Lab4\zad4_disparity.png"
    output_path_Zad4_depth = r"C:\#Projects\ZAOWIR\Lab4\zad4_depth.png"

    h_fov = 60
    image_width = 1920
    baseline = 0.1  #meters
    max_depth = 1000.0
    depth_map_uint24 = cv2.imread(r"C:\#Projects\ZAOWIR\Lab4\depth.png", cv2.IMREAD_UNCHANGED)
    focal_length = (depth_map_uint24.shape[0] / 2) / np.tan(np.radians(h_fov) / 2)

    depth_map_zad4 = decode_uint24_depth_map(depth_map_uint24, max_depth)
    disparity_map_zad4 = depth_to_disparity(depth_map_zad4, baseline, focal_length)

    cv2.imwrite(output_path_Zad4_disparity, disparity_map_zad4)
    cv2.imwrite(output_path_Zad4_depth, depth_map_zad4)

    image = cv2.cvtColor(depth_map_uint24, cv2.COLOR_BGR2RGB)
    pixels = np.array(image)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(pixels)
    plt.title('Reference Depth Map 24-bit')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(depth_map_zad4, cmap='gray')
    plt.title('Reference Depth Map Normalized')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(disparity_map_zad4, cmap='gray')
    plt.title('Decoded Disparity Map')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # Zad 5
    input_file = r"C:\#Projects\ZAOWIR\Lab4\left.png"
    output_path_Zad5_ply = r"C:\#Projects\ZAOWIR\Lab4\zad5.ply"

    input_file_Zad5_1 = cv2.imread(input_file, 0)
    disparity_map_zad4_8bit = cv2.imread(output_path_Zad4_disparity, 0)
    depth_map_zad4 = cv2.imread(output_path_Zad4_depth, 0)
    h, w = input_file_Zad5_1.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],
                    [0, 0, 0, -f],
                    [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disparity_map_zad4_8bit, Q)
    colors = cv2.cvtColor(input_file_Zad5_1, cv2.COLOR_BGR2RGB)
    mask = depth_map_zad4 < 50
    out_points = points[mask]
    out_colors = colors[mask]
    write_ply(output_path_Zad5_ply, out_points, out_colors)