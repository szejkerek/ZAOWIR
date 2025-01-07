import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define error metrics functions without sklearn
def calculate_mae(gt, pred):
    return np.mean(np.abs(gt.astype(np.float32) - pred.astype(np.float32)))

def calculate_rmse(gt, pred):
    return np.sqrt(np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2))

def calculate_bad_pixel(gt, pred, threshold=1):
    return np.mean(np.abs(gt.astype(np.float32) - pred.astype(np.float32)) > threshold) * 100

def calculate_ssim(gt, pred):
    C1 = 6.5025
    C2 = 58.5225
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    mu_gt = cv2.GaussianBlur(gt, (11, 11), 1.5)
    mu_pred = cv2.GaussianBlur(pred, (11, 11), 1.5)
    mu_gt_sq = mu_gt ** 2
    mu_pred_sq = mu_pred ** 2
    mu_gt_pred = mu_gt * mu_pred
    sigma_gt_sq = cv2.GaussianBlur(gt ** 2, (11, 11), 1.5) - mu_gt_sq
    sigma_pred_sq = cv2.GaussianBlur(pred ** 2, (11, 11), 1.5) - mu_pred_sq
    sigma_gt_pred = cv2.GaussianBlur(gt * pred, (11, 11), 1.5) - mu_gt_pred
    ssim_map = ((2 * mu_gt_pred + C1) * (2 * sigma_gt_pred + C2)) / \
               ((mu_gt_sq + mu_pred_sq + C1) * (sigma_gt_sq + sigma_pred_sq + C2))
    return ssim_map.mean()

def visualize_error_map(gt, pred, title="Error Map"):
    error_map = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
    plt.figure(figsize=(10, 5))
    plt.imshow(error_map, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.close()

def save_disparity_map(disparity_map, filename):
    cv2.imwrite(filename, disparity_map)

def compute_custom_disparity(left_img, right_img, block_size=5, disparity_range=64):
    if block_size % 2 == 0:
        block_size += 1
    half_block_size = block_size // 2
    height, width = left_img.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    for y in tqdm(range(half_block_size, height - half_block_size), desc="Processing Rows", unit="row"):
        for x in range(half_block_size, width - half_block_size):
            left_block = left_img[y - half_block_size:y + half_block_size + 1,
                                  x - half_block_size:x + half_block_size + 1]
            min_sad = float('inf')
            best_disparity = 0
            for disparity in range(0, disparity_range):
                if x - disparity - half_block_size < 0:
                    continue
                right_block = right_img[y - half_block_size:y + half_block_size + 1,
                                        x - disparity - half_block_size:x - disparity + half_block_size + 1]
                if right_block.shape != left_block.shape:
                    continue
                sad = np.sum(np.abs(left_block.astype(np.float32) - right_block.astype(np.float32)))
                if sad < min_sad:
                    min_sad = sad
                    best_disparity = disparity
            disparity_map[y, x] = best_disparity
    disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)
    return disparity_map_normalized


left_img = cv2.imread("C:\\#Projects\\ZAOWIR\\Lab3\\GordonKolokwium\\Xleft.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("C:\\#Projects\\ZAOWIR\\Lab3\\GordonKolokwium\\Xright.png", cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread("C:\\#Projects\\ZAOWIR\\Lab3\\GordonKolokwium\\ground_truth.png", cv2.IMREAD_GRAYSCALE)

if left_img is None or right_img is None or ground_truth is None:
    raise ValueError("Error loading images")


left_img = cv2.equalizeHist(left_img)
right_img = cv2.equalizeHist(right_img)

disparity_custom = compute_custom_disparity(left_img, right_img)

stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=11)
disparity_bm = stereo_bm.compute(left_img, right_img)
disparity_bm = cv2.normalize(disparity_bm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=150,
    speckleRange=32
)
disparity_sgbm = stereo_sgbm.compute(left_img, right_img)
disparity_sgbm = cv2.normalize(disparity_sgbm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


save_disparity_map(disparity_custom, "disparity_custom.png")
save_disparity_map(disparity_bm, "disparity_bm.png")
save_disparity_map(disparity_sgbm, "disparity_sgbm.png")

print("Custom MAE:", calculate_mae(ground_truth, disparity_custom))
print("Custom RMSE:", calculate_rmse(ground_truth, disparity_custom))
print("Custom Bad Pixel %:", calculate_bad_pixel(ground_truth, disparity_custom))
print("Custom SSIM:", calculate_ssim(ground_truth, disparity_custom))
visualize_error_map(ground_truth, disparity_custom, "Custom Error Map")

print("BM MAE:", calculate_mae(ground_truth, disparity_bm))
print("BM RMSE:", calculate_rmse(ground_truth, disparity_bm))
print("BM Bad Pixel %:", calculate_bad_pixel(ground_truth, disparity_bm))
print("BM SSIM:", calculate_ssim(ground_truth, disparity_bm))
visualize_error_map(ground_truth, disparity_bm, "BM Error Map")

print("SGBM MAE:", calculate_mae(ground_truth, disparity_sgbm))
print("SGBM RMSE:", calculate_rmse(ground_truth, disparity_sgbm))
print("SGBM Bad Pixel %:", calculate_bad_pixel(ground_truth, disparity_sgbm))
print("SGBM SSIM:", calculate_ssim(ground_truth, disparity_sgbm))
visualize_error_map(ground_truth, disparity_sgbm, "SGBM Error Map")
