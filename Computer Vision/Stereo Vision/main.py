import cv2
import numpy as np
import matplotlib.pyplot as plt

def drawlines(img1, img2, lines):
    r, c = img1.shape[:2]
    img1_lines = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    img2_lines = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()

    for line in lines:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        img1_lines = cv2.line(img1_lines, (x0, y0), (x1, y1), color, 3)  

    return img1_lines, img2_lines

def match_epipolar_lines(linesL, linesR, left_gray, right_gray, window_size=15):
    matches = []
    rows, cols = left_gray.shape
    for lineL, lineR in zip(linesL, linesR):
        aL, bL, cL = lineL
        aR, bR, cR = lineR
        x_left = np.linspace(0, cols, num=100, dtype=np.int32)
        y_left = np.int32(-(aL * x_left + cL) / bL)
        for x, y in zip(x_left, y_left):
            if 0 <= y < rows and 0 <= x < cols:
                window_left = left_gray[max(y - window_size // 2, 0):min(y + window_size // 2, rows),
                                   max(x - window_size // 2, 0):min(x + window_size // 2, cols)]
                best_match_x = None
                min_ssd = float('inf')
                for x_r in range(0, cols, window_size):
                    y_r = np.int32(-(aR * x_r + cR) / bR)
                    if 0 <= y_r < rows:
                        window_right = right_gray[max(y_r - window_size // 2, 0):min(y_r + window_size // 2, rows),
                                             max(x_r - window_size // 2, 0):min(x_r + window_size // 2, cols)]
                        if window_right.shape == window_left.shape:
                            ssd = np.sum((window_left - window_right) ** 2)
                            if ssd < min_ssd:
                                min_ssd = ssd
                                best_match_x = x_r
                if best_match_x is not None:
                    matches.append((x, y, best_match_x))
    return matches

def draw_matches(img1, img2, matches):
    img_matches = np.hstack((img1, img2))
    for (x1, y1, x2) in matches:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        y2 = int(y1)
        img_matches = cv2.line(img_matches, (int(x1), int(y1)), (int(x2) + img1.shape[1], y2), color, 1)
        img_matches = cv2.circle(img_matches, (int(x1), int(y1)), 5, color, -1)
        img_matches = cv2.circle(img_matches, (int(x2) + img1.shape[1], y2), 5, color, -1)
    return img_matches

def rectify_images(imgL, imgR):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        preFilterCap=4,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(imgL, imgR)

    return disparity



def normalize_disparity(disparity):
    disparity = np.clip(disparity, 0, disparity.max())
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    disparity = (disparity * 255).astype(np.uint8)
    return disparity

def compute_depth_map(disparity_map, focal_length, baseline):
    disparity_map[disparity_map == 0] = 0.1

    depth_map = (focal_length * baseline) / disparity_map
    
    depth_map[np.isinf(depth_map)] = 0
    depth_map[np.isnan(depth_map)] = 0

    return depth_map

def enhance_depth_map(depth_map):
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.clip(depth_map_normalized, 0, 255)  
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    contrast = cv2.convertScaleAbs(depth_map_normalized, alpha=1.5, beta=0)
    return contrast

def main():
    left = cv2.imread("Dataset1/im1.png")
    right = cv2.imread("Dataset1/im0.png")    

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypointsL, descriptorsL = sift.detectAndCompute(left_gray, None)
    keypointsR, descriptorsR = sift.detectAndCompute(right_gray, None)

    cornersL = cv2.goodFeaturesToTrack(left_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    cornersR = cv2.goodFeaturesToTrack(right_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    cornersL_kp = [cv2.KeyPoint(c[0][0], c[0][1], 1) for c in cornersL]
    cornersR_kp = [cv2.KeyPoint(c[0][0], c[0][1], 1) for c in cornersR]
    keypointsL_combined = list(keypointsL) + cornersL_kp
    keypointsR_combined = list(keypointsR) + cornersR_kp
    
    _, descriptorsL_corners = sift.compute(left_gray, cornersL_kp)
    _, descriptorsR_corners = sift.compute(right_gray, cornersR_kp)
    descriptorsL_combined = np.vstack((descriptorsL, descriptorsL_corners))
    descriptorsR_combined = np.vstack((descriptorsR, descriptorsR_corners))
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptorsL_combined, descriptorsR_combined)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]
    
    left_points = np.float32([keypointsL_combined[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    right_points = np.float32([keypointsR_combined[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    F, mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC)
    
    cam0 = np.array([[5299.313  , 0, 1263.818],
                     [0, 5299.313,977.763],
                     [0, 0, 1]])
    cam1 = np.array([[5299.313  , 0, 1438.004],
                     [0, 5299.313,977.763],
                     [0, 0, 1]])
    E = np.dot(np.dot(cam0.T, F), cam1)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    _, HL, HR = cv2.stereoRectifyUncalibrated(left_points.reshape(-1, 2), right_points.reshape(-1, 2), F, imgSize=(left_gray.shape[1], left_gray.shape[0]))
    left_rectified = cv2.warpPerspective(left_gray, HL, (left_gray.shape[1], left_gray.shape[0]))
    right_rectified = cv2.warpPerspective(right_gray, HR, (right_gray.shape[1], right_gray.shape[0]))
    
    linesL = cv2.computeCorrespondEpilines(right_points.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    linesR = cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    
    left_with_lines, _ = drawlines(left, right_rectified, linesL)
    right_with_lines, _ = drawlines(right, left_rectified, linesR)

    disparity = rectify_images(left_gray, right_gray)
    disparity_normalized = normalize_disparity(disparity)

    disparity_heatmap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    focal_length =5299.313  
    baseline = 177.288

    depth_map = compute_depth_map(disparity, focal_length, baseline)

    depth_map_enhanced = enhance_depth_map(depth_map)
    depth_heatmap = cv2.applyColorMap(depth_map_enhanced, cv2.COLORMAP_SUMMER)

    plt.figure(figsize=(24, 12))

    plt.subplot(241)
    plt.imshow(cv2.cvtColor(left_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Left Image with Epipolar Lines')
    plt.axis('off')

    plt.subplot(242)
    plt.imshow(cv2.cvtColor(right_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Right Image with Epipolar Lines')
    plt.axis('off')

    plt.subplot(243)
    plt.imshow(depth_map_enhanced, cmap='gray')
    plt.title('Depth Map')
    plt.axis('off')

    plt.subplot(245)
    plt.imshow(disparity_normalized, cmap='gray')
    plt.title('Disparity Map')
    plt.axis('off')

    plt.subplot(246)
    plt.imshow(cv2.cvtColor(disparity_heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Disparity Heatmap')
    plt.axis('off')

    plt.subplot(247)
    plt.imshow(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Depth Heatmap')
    plt.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.show()

if __name__ == "__main__":
    main()
