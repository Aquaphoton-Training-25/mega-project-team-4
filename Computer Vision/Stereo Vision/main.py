import cv2
import numpy as np
import matplotlib.pyplot as plt

def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[:2]
    img1_lines = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_lines = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        img1_lines = cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)        
        pt1_int = tuple(map(int, pt1[0]))
        pt2_int = tuple(map(int, pt2[0]))
        img1_lines = cv2.circle(img1_lines, pt1_int, 5, color, -1)
        img2_lines = cv2.circle(img2_lines, pt2_int, 5, color, -1)
    return img1_lines, img2_lines

def match_epipolar_lines(linesL, linesR, left, right, window_size=15):
    matches = []
    rows, cols = left.shape
    for lineL, lineR in zip(linesL, linesR):
        aL, bL, cL = lineL
        aR, bR, cR = lineR
        x_left = np.linspace(0, cols, num=100, dtype=np.int32)
        y_left = np.int32(-(aL * x_left + cL) / bL)
        for x, y in zip(x_left, y_left):
            if 0 <= y < rows and 0 <= x < cols:
                window_left = left[max(y - window_size // 2, 0):min(y + window_size // 2, rows),
                                   max(x - window_size // 2, 0):min(x + window_size // 2, cols)]              
                best_match_x = None
                min_ssd = float('inf')
                for x_r in range(0, cols, window_size):
                    y_r = np.int32(-(aR * x_r + cR) / bR)
                    if 0 <= y_r < rows:
                        window_right = right[max(y_r - window_size // 2, 0):min(y_r + window_size // 2, rows),
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

def compute_disparity_map(left_points, right_points, img_shape):
    disparity_map = np.zeros(img_shape[:2], dtype=np.float32)
    
    for pt1, pt2 in zip(left_points, right_points):
        x1, y1 = pt1[0]
        x2, y2 = pt2[0]

        # Ensure the points are within the image boundaries
        if 0 <= int(y1) < img_shape[0] and 0 <= int(x1) < img_shape[1] and 0 <= int(y2) < img_shape[0] and 0 <= int(x2) < img_shape[1]:
            disparity = x1 - x2  # Disparity as the difference in x-coordinates
            if disparity > 0:  # Only use positive disparity values
                disparity_map[int(y1), int(x1)] = disparity
                
    return disparity_map


def main():
    left = cv2.imread("Dataset1/im1.png")
    right = cv2.imread("Dataset1/im0.png")    
    
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    leftCopy=left_gray
    rightCopy=right_gray
    
    # SIFT feature detection and matching
    sift = cv2.SIFT_create()
    keypointsL, descriptorsL = sift.detectAndCompute(left_gray, None)
    keypointsR, descriptorsR = sift.detectAndCompute(right_gray, None)
    
    # Corner detection and combining with SIFT features
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
    
    # Matching with BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptorsL_combined, descriptorsR_combined)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]
    
    # Extracting matched points
    left_points = np.float32([keypointsL_combined[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    right_points = np.float32([keypointsR_combined[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Fundamental matrix
    F, mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_RANSAC)
    
    # Camera matrices (replace with actual values)
    cam0 = np.array([[5299.313, 0, 1263.818],
                     [0, 5299.313 , 977.763],
                     [0, 0, 1]])
    cam1 = np.array([[4396.869, 0, 1438.004],
                     [0, 4396.869,989.702],
                     [0, 0, 1]])
    
    # Essential matrix and decomposition
    E = np.dot(np.dot(cam0.T, F), cam1)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    # Rectification
    _, HL, HR = cv2.stereoRectifyUncalibrated(left_points.reshape(-1, 2), right_points.reshape(-1, 2), F, imgSize=(left_gray.shape[1], left_gray.shape[0]))
    left_rectified = cv2.warpPerspective(left_gray, HL, (left_gray.shape[1], left_gray.shape[0]))
    right_rectified = cv2.warpPerspective(right_gray, HR, (right_gray.shape[1], right_gray.shape[0]))
    
    # Epipolar lines
    linesL = cv2.computeCorrespondEpilines(right_points.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    linesR = cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    
    left_with_lines, _ = drawlines(left_rectified, right_rectified, linesL, left_points, right_points)
    right_with_lines, _ = drawlines(right_rectified, left_rectified, linesR, right_points, left_points)
    
    # Match epipolar lines
    matches = match_epipolar_lines(linesL, linesR, left_rectified, right_rectified)
    matches_img = draw_matches(left_rectified, right_rectified, matches)
    
    left_points_rectified = cv2.perspectiveTransform(left_points, HL)
    right_points_rectified = cv2.perspectiveTransform(right_points, HR)
    
    # # Compute the disparity map
    # disparity_map = compute_disparity_map(left_points_rectified, right_points_rectified, left_gray.shape)
    
    # # Normalize the disparity map for visualization
    # if disparity_map.max() > 0:
    #     disparity_map = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min())
    #     disparity_map = (disparity_map * 255).astype(np.uint8)
    # else:
    #     print("Disparity map has no valid values.")
    # Display results
 # Parameters for StereoBM
    numDisparities = 16  # Number of disparities to search (must be divisible by 16)
    blockSize = 15  # Matched block size, the larger it is, the smoother the disparity map
    
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(leftCopy, rightCopy).astype(np.float32) / 16.0

    # Normalize the disparity map for visualization
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    disparity = (disparity * 255).astype(np.uint8)
    



    plt.figure(figsize=(18, 12))
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(left_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Left Image with Epipolar Lines')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(right_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Right Image with Epipolar Lines')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(left_rectified)
    plt.title('Matches along Epipolar Lines')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(disparity, cmap='gray')
    plt.title('Disparity Map')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

if __name__ == "__main__":
    main()
