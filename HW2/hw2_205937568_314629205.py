import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dean Amar 205937568
# Liran Eliav 314629205

class ProduceProjection:
    def __init__(self, image, K, depthMap, baseLine = 10):
        self.imageToCamera = {}
        self.baseLine = baseLine
        self.K = K
        self.inverseK = np.linalg.inv(self.K)
        self.extr = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        self.extr = np.array(self.extr)
        self.P = self.K @ self.extr

        self.depthMap = depthMap
        self.FX, self.Ox, self.FY, self.Oy, *_ = self.extractInnerParamertsFromK(K)
        self.row, self.col, *_ = image.shape
        self.leftImage = image

    def getDepth(self, i, j):
        return self.depthMap[i, j]

    def mapImageToCamera(self):
        for i in range(self.row):
            for j in range(self.col):
                xyz = self.inverseK @ np.array([i,j,1]) * self.getDepth(i,j)
                self.imageToCamera[(i, j)] = (xyz[0], xyz[1], xyz[2], self.leftImage[i, j])

    def reprojectImage(self):
        self.reprojectCor = {}
        reconstructedImage = np.zeros((self.row, self.col, 3))
        for coordinate in self.imageToCamera:
            x, y, z, val = self.imageToCamera[coordinate]
            cameraVec = np.array([x,y,z,1])
            resVec = self.P@cameraVec.T
            if z != 0:
                resVec = resVec/z
            else:
                resVec = resVec

            self.reprojectCor[(int(resVec[0]), int(resVec[1]))] = val

        for i in range(self.row):
            for j in range(self.col):
                if (i, j) in self.reprojectCor.keys():
                    reconstructedImage[i, j] = self.reprojectCor[(i, j)]
                else:
                    reconstructedImage[i, j] = 0

        return reconstructedImage


    def tuneP(self, unit):
        if unit == 0:
            unit = 0
        self.extr = [[1,0,0,0],[0,1,0,-unit],[0,0,1,0]]
        self.extr = np.array(self.extr, dtype='f')

        self.P = self.K @ self.extr

    def extractInnerParamertsFromK(self, K):
        listOfParameters = K.flatten()
        listOfParameters = listOfParameters[listOfParameters != 0] #1-> FX, 2-> Ox, 3-> fy, 4 -> Oy

        return tuple(listOfParameters)

def XOR(list1, list2):
    non_similar = 0
    for i in range(len(list1)):
        if list1[i] != list2[i]:
            non_similar = non_similar + 1

    return non_similar

class DisparityMap:
    def __init__(self):
        pass

    def makeCensusTransform(self, image, filterSize):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        row1, col1 = image.shape
        censusTransformMat = np.zeros((row1, col1, 1))

        val = int((filterSize)/2)
        paddedImage = np.pad(image, ((val, val),(val, val)),'constant', constant_values = 0)

        row, col = paddedImage.shape

        for i in range(val, row-val):
            for j in range(val, col-val):
                pixelValue = paddedImage[i, j]
                hammingValues = ""
                for x in range(i-val, i+val+1):
                    for y in range(j-val, j+val+1):
                        if (x == i) & (y == j):
                            continue
                        if paddedImage[x, y] >= pixelValue:
                            hammingValues += "1"
                        else:
                            hammingValues += "0"

                censusTransformMat[i-val, j-val] = int(hammingValues, 2)

        return censusTransformMat
    

    def makeCostVolume(self, censusTrasformPic1, censusTrasformPic2, path, isRight = None):
        # maxDisparity = int(open(path, "r").read())
        maxDisparity = 31
        row, col, diff = np.array(censusTrasformPic1).shape

        costVolume = np.zeros((maxDisparity, row, col)) + 255

        for i in range(row):
            for j in range(0, col):
                for k in range(0, maxDisparity):
                    if isRight:
                        if j + k >= col:
                            break
                        binary_string = bin(int(censusTrasformPic2[i, j]) ^ int(censusTrasformPic1[i, j + k]))
                        holder = binary_string.count('1')
                        costVolume[k, i, j] = holder

                    else:
                        if j - k < 0:
                            break
                        binary_string = bin(int(censusTrasformPic1[i, j]) ^ int(censusTrasformPic2[i, j - k]))
                        holder = binary_string.count('1')
                        costVolume[k, i, j] = holder

        return costVolume


    def makeLocalAggregation(self, costVolume, windowSize):
        costVolumeAgg = costVolume.copy()
        maxDis, row, col = costVolume.shape
        for i in range(maxDis):
            costVolumeAgg[i] = cv2.blur(costVolumeAgg[i], (windowSize, windowSize))

        return costVolumeAgg

    def findMinimumForEachLocal(self, costVolumeAgg):
        disparityDim, row, col = costVolumeAgg.shape
        minCostVolume = np.zeros((row, col))

        for i in range(row):
            for j in range(col):
                dispValues = []
                for disp in range(0, disparityDim):
                    dispValues.append(costVolumeAgg[disp][i][j])
                minCostVolume[i, j] = dispValues.index(min(dispValues))

        return minCostVolume


    def consistencyCheck(self, costVolLeft, costVolRight, LR = True):
        if LR == True:
            mulFactor = -1
        else:
            mulFactor = 1

        row, col, *_ = costVolLeft.shape
        matchMat = np.zeros((row,col))
        for i in range(row):
            for j in range(col):
                k = int(costVolLeft[i][j])
                if j + k < col:
                    if costVolLeft[i][j] == costVolRight[i][j+k*mulFactor]:
                        matchMat[i][j] = costVolLeft[i][j]
                    else:
                        matchMat[i][j] = 0

        return matchMat

    def consistencyCheckRightToLeft(self, costVolLeft, costVolRight):
        row, col, *_ = costVolRight.shape
        matchMat = np.zeros((row,col))
        for i in range(row):
            for j in range(col):
                k = int(costVolRight[i][j])
                if j + k < col:
                    if costVolRight[i][j] == costVolLeft[i][j+k]:
                        matchMat[i][j] = costVolRight[i][j]
                    else:
                        matchMat[i][j] = 0

        return matchMat

def create_depth_map(disparity_map, baseline=10, focal_length=5.76):

    row, col = disparity_map.shape
    #depth_map = np.zeros_like(disparity_map)
    depth_map = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if disparity_map[i,j] != 0:
                depth_map[i,j] = (focal_length) * (baseline / disparity_map[i,j])
            else:
                depth_map[i,j] = 0

    return depth_map

def ignore_8_maxes(depth_map):
    row, col = depth_map.shape

    for m in range(8):
        maximum = depth_map.max()
        for i in range(row):
            for j in range(col):
                if depth_map[i,j] == maximum:
                    depth_map[i,j] = 0

    return depth_map

def generate_point_cloud(disparity_map, depth_map, baseline, focal_length):
    rows, cols , dep= disparity_map.shape
    point_cloud = []

    for y in range(rows):
        for x in range(cols):
            disparity = disparity_map[y, x]
            depth = depth_map[y, x]
            if disparity > 0 and depth > 0:
                z = (baseline * focal_length) / disparity
                x_3d = (x - cols/2) * (z / focal_length)
                y_3d = (y - rows/2) * (z / focal_length)
                point_cloud.append([x_3d, y_3d, z])

    return np.array(point_cloud)



if __name__ == '__main__':
    os.makedirs('example_Outputs', exist_ok=True)
    os.makedirs('set1_Outputs', exist_ok=True)
    os.makedirs('set2_Outputs', exist_ok=True)
    os.makedirs('set3_Outputs', exist_ok=True)
    os.makedirs('set4_Outputs', exist_ok=True)
    os.makedirs('set5_Outputs', exist_ok=True)

    disparityMaker = DisparityMap()
    # path = "Data/example/rot_low.h264"
    # cap = cv2.VideoCapture(path)
    # for i in range(30):
        # cap.read()
    # ret1,leftImage = cap.read()
    # leftImage = cv2.rotate(leftImage, cv2.ROTATE_90_CLOCKWISE)
    # leftImage = leftImage[250:400,0:400]
    # for i in range(10):
        # cap.read()
    # ret2,rightImage = cap.read()
    # rightImage = cv2.rotate(rightImage, cv2.ROTATE_90_CLOCKWISE)
    # rightImage = rightImage[250:400,0:400]
    # cv2.imrea('example_Outputs\\im_left.jpg', leftImage)
    # cv2.imwrite('example_Outputs\\im_right.jpg', rightImage)

    if 1:
        print('---example---')
        print("Step one")
        #censusTransLeft = disparityMaker.makeCensusTransform(leftImage, 5)
        censusTransLeft = disparityMaker.makeCensusTransform(leftImage, 7)
        #censusTransRight = disparityMaker.makeCensusTransform(rightImage, 5)
        censusTransRight = disparityMaker.makeCensusTransform(rightImage, 7)

        print("Step Two")

        max_disp_path = "Data/example/max_disp.txt"

        costVolumeRL = disparityMaker.makeCostVolume(censusTransLeft, censusTransRight, max_disp_path, True)
        costVolumeLR = disparityMaker.makeCostVolume(censusTransLeft, censusTransRight, max_disp_path, False)

        print("Step Three")

        aggCostVolumeLR = disparityMaker.makeLocalAggregation(costVolumeLR, 17)
        aggCostVolumeRL = disparityMaker.makeLocalAggregation(costVolumeRL, 17)

        print("Step Four")

        minimumCostVolumeLR = disparityMaker.findMinimumForEachLocal(aggCostVolumeLR)
        minimumCostVolumeRL = disparityMaker.findMinimumForEachLocal(aggCostVolumeRL)

        LastLR = disparityMaker.consistencyCheck(minimumCostVolumeLR, minimumCostVolumeRL, True)
        LastRL = disparityMaker.consistencyCheck(minimumCostVolumeRL, minimumCostVolumeLR, False)

        # np.savetxt('example_Outputs\\disparty-map-left.txt', LastLR, delimiter=',', fmt='%d')
        # np.savetxt('example_Outputs\\disparty-map-right.txt', LastRL, delimiter=',', fmt='%d')

        K = np.loadtxt("Data/example/K.txt", dtype='f', delimiter='\t')
        focal_length = K[0,0] / 100

        depth_map_leftright = create_depth_map(LastLR, 1, focal_length)
        depth_map_rightleft = create_depth_map(LastRL, 1, focal_length)

        # print('Step Five')

        # # np.savetxt('example_Outputs\\depth-map-left-right.txt', depth_map_leftright, delimiter=',', fmt='%f')
        # # np.savetxt('example_Outputs\\depth-map-right-left.txt', depth_map_rightleft, delimiter=',', fmt='%f')

        print("Last Step")

        projectionClass = ProduceProjection(leftImage, K, depth_map_leftright)

        for i in range(11):
            projectionClass.tuneP(i * 0.01)
            projectionClass.mapImageToCamera()
            image = projectionClass.reprojectImage()
            # cv2.imwrite(f'example_Outputs\\synth{i+1}.jpg', image)

        lastLR_max = LastLR.max()
        if lastLR_max != 0:
            LastLR = LastLR / lastLR_max

        lastRL_max = LastRL.max()
        if lastRL_max != 0:
            LastRL = LastRL / lastRL_max

        # cv2.imwrite('example_Outputs\\disp_left.jpg', (LastLR * 255).astype(np.uint8))
        # cv2.imwrite('example_Outputs\\disp_right.jpg', (LastRL * 255).astype(np.uint8))

        depth_map_leftright = ignore_8_maxes(depth_map_leftright)

        depth_map_leftright_max = depth_map_leftright.max()
        if depth_map_leftright_max != 0:
            depth_map_leftright = depth_map_leftright / depth_map_leftright_max

        depth_map_rightleft = ignore_8_maxes(depth_map_rightleft)

        depth_map_rightleft_max = depth_map_rightleft.max()
        if depth_map_rightleft_max != 0:
            depth_map_rightleft = depth_map_rightleft / depth_map_rightleft_max

        cv2.imwrite('example_Outputs\\depth_left.jpg', (depth_map_leftright * 255).astype(np.uint8))
        cv2.imwrite('example_Outputs\\depth_right.jpg', (depth_map_rightleft * 255).astype(np.uint8))
        
        point_cloud = generate_point_cloud(np.array(LastLR),np.array(depth_map_leftright),-10,5.062113516171736)
    
        # After generating the depth_left and depth_right matrices

        # # Set your depth range thresholds for the woman's region
        # woman_depth_lower = 100  # Adjust this based on your scene
        # woman_depth_upper = 300  # Adjust this based on your scene

        # # Create a mask for points within the woman's depth range
        # woman_mask = (depth_left >= woman_depth_lower) & (depth_left <= woman_depth_upper)

        # Create a copy of your point_cloud
        woman_point_cloud = point_cloud.copy()

        # Apply the mask to keep only the woman's points
        # woman_point_cloud = woman_point_cloud[woman_mask.flatten()]

        min_xyz = woman_point_cloud.min(axis=0)
        max_xyz = woman_point_cloud.max(axis=0)
        scaled_woman_point_cloud = 400 + (woman_point_cloud - min_xyz) * (500 - 100) / (max_xyz-min_xyz)

        # Visualize the point cloud with only the woman's points in white
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(scaled_woman_point_cloud[:, 0], scaled_woman_point_cloud[:, 1], scaled_woman_point_cloud[:, 2], s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        

        print("Finished Running")