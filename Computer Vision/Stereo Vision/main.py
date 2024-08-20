import cv2
import numpy as np
import matplotlib.pyplot as plt
class StereoCamera:
    def __init__(self):
        self.cam_matrix_left = np.array([[5806.559, 0, 1429.219],
                                         [0, 5806.559, 993.403],
                                         [0, 0, 1]])
        
        self.cam_matrix_right = np.array([[5806.559, 0, 1543.51],
                                          [0, 5806.559, 993.403],
                                          [0, 0, 1]])
        self.distortion_l = np.zeros(5)  
        self.distortion_r = np.zeros(5)  
        self.R = np.eye(3)  
        self.T = np.array([[174.019, 0, 0]]).T  
        self.doffs = 114.291  

    def setMiddleBurryParams_Adirondack(self):
        pass

def preprocess(img1, img2):
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
    return img1, img2

def undistortion(image, camera_matrix, dist_coeff):
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeff)
    return undistorted_image

def getRectifyTransform(height, width, config):
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T
    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, Q

def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectified_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectified_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectified_img1, rectified_img2


def stereoMatchSGBM(left_image, right_image, down_scale=False):
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 128,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml.copy()
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)
    size = (left_image.shape[1], left_image.shape[0])
    if not down_scale:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]
        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right
    trueDisp_left = disparity_left.astype(np.float32) / 16
    trueDisp_right = disparity_right.astype(np.float32) / 16
    return trueDisp_left, trueDisp_right

def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.array) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q, handleMissingValues=False, ddepth=-1)
    depthMap = points_3d[:, :, 2]
    depthMap = np.where(np.abs(depthMap) < 1e-5, 0, depthMap) 
    return depthMap.astype(np.float32)

def getDepthMapWithConfig(disparityMap: np.ndarray, config: StereoCamera) -> np.ndarray:
    focal_length_px = config.cam_matrix_left[0, 0] 
    baseline_mm = config.T[0, 0]  
    baseline_m = baseline_mm / 1000.0  
    doffs = config.doffs  
    disparityMap = np.clip(disparityMap, 1e-5, None)
    depthMap = (focal_length_px * baseline_m) / (disparityMap + doffs)  
    depthMap = np.clip(depthMap, 0, 10000)  
    return depthMap.astype(np.float32)
def getRealWorldCoords(x, y, depthMap, config):
    Z = depthMap[y, x]     
    if Z <= 0: 
        return np.array([0.0, 0.0, 0.0])    
    X = (x - config.cam_matrix_left[0, 2]) * Z / config.cam_matrix_left[0, 0]
    Y = (y - config.cam_matrix_left[1, 2]) * Z / config.cam_matrix_left[1, 1]
    return np.array([X, Y, Z])
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(point1) == 0:
            point1.append(x)
            point1.append(y)
            print("Selected point1:", point1)
        elif len(point2) == 0:
            point2.append(x)
            point2.append(y)
            print("Selected point2:", point2)
            original_point1 = [int(point1[0] / scale_factor), int(point1[1] / scale_factor)]
            original_point2 = [int(point2[0] / scale_factor), int(point2[1] / scale_factor)]
            coords1 = getRealWorldCoords(original_point1[0], original_point1[1], depthMap, config)
            coords2 = getRealWorldCoords(original_point2[0], original_point2[1], depthMap, config)
            distance_meters = np.linalg.norm(coords1 - coords2)/3
            print(f"Distance between points: {distance_meters:.4f} meters")
def visualizeDepthMap(depthMap: np.ndarray) -> np.ndarray:
    minDepth = np.min(depthMap[depthMap > 0])  
    maxDepth = np.max(depthMap)
    depthMapVis = (255.0 * (maxDepth - depthMap)) / (maxDepth - minDepth)
    depthMapVis = np.clip(depthMapVis, 0, 255).astype(np.uint8)
    return depthMapVis

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

if __name__ == '__main__':
    left = cv2.imread('Dataset1/im0.png')
    right = cv2.imread('Dataset1/im1.png')
    height, width = left.shape[0:2]

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypointsL, descriptorsL = sift.detectAndCompute(left, None)
    keypointsR, descriptorsR = sift.detectAndCompute(right, None)

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
    
    _, HL, HR = cv2.stereoRectifyUncalibrated(left_points.reshape(-1, 2), right_points.reshape(-1, 2), F, imgSize=(left_gray.shape[1], left_gray.shape[0]))
    left_rectified = cv2.warpPerspective(left_gray, HL, (left_gray.shape[1], left_gray.shape[0]))
    right_rectified = cv2.warpPerspective(right_gray, HR, (right_gray.shape[1], right_gray.shape[0]))

    print("the left homography matrix")
    print(HL)
    print("the right homography matrix")
    print(HR)
    
    linesL = cv2.computeCorrespondEpilines(right_points.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    linesR = cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    left_with_lines, _ = drawlines(left, right_rectified, linesL)
    right_with_lines, _ = drawlines(right, left_rectified, linesR)

    config = StereoCamera()
    config.setMiddleBurryParams_Adirondack()
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    left_rectified, right_rectified = rectifyImage(left, right, map1x, map1y, map2x, map2y)
    left_, right_ = preprocess(left, right)
    disparity, _ = stereoMatchSGBM(left_, right_, False)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)
    cv2.imwrite('disparity_greyscale.png', disparity)
    disparity_heatmap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_WINTER)
    cv2.imwrite('disparity_heatmap.png', disparity_heatmap)

    depthMap = getDepthMapWithConfig(disparity, config)
    depthMapVis = visualizeDepthMap(depthMap)
   
    cv2.imwrite('depth.png', depthMapVis)
    depth_heatmap = cv2.applyColorMap(depthMapVis, cv2.COLORMAP_JET)
    cv2.imwrite('depth_heatmap.png', depth_heatmap)  
    cv2.imwrite('depth_greyscale.png', depthMapVis) 
    screen_height = 1080  
    scale_factor = height / screen_height
    display_depth_map = cv2.resize(depthMapVis, (int(width / scale_factor), screen_height))    

    plt.figure(figsize=(16, 9))  

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(left_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Left Image with Epipolar Lines')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(right_with_lines, cv2.COLOR_BGR2RGB))
    plt.title('Right Image with Epipolar Lines')
    plt.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.tight_layout()
    plt.show() 

    point1, point2 = [], []
    cv2.imshow("depth map", display_depth_map)
    cv2.setMouseCallback("depth map", select_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
