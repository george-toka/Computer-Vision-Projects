import numpy as np
import cv2 as cv

def MedianFilter(src,windowSize):

    k = windowSize
    bounds = int(k/2)
    MedianIndex = int(k*k/2)

    #no padding
    h, w = len(src),len(src[0])
    Out = np.zeros((h, w))  

    #Median Blur
    for i in range(bounds,h-bounds):
        for j in range(bounds,w-bounds): 
            temp = np.zeros((k*k))         
            for u in range(k):
                for v in range(k):
                    temp[u*k + v] = src[i+u-bounds][j+v-bounds]
            Out[i][j] = np.sort(temp)[MedianIndex]

    #Copy values of neighboring pixels to border
    #First & Last Column
    for i in range(1,h-1):
        Out[i][0] = Out[i][1]
        Out[i][w-1] = Out[i][w-2]

    #First & Last Row 
    for j in range(1,w-1):
        Out[0][j] = Out[1][j]
        Out[h-1][j] = Out[h-2][j]

    #Diagonals
    Out[0][0] = Out[1][1]
    Out[h-1][0] = Out[h-2][1]
    Out[0][w-1] = Out[1][w-2]
    Out[h-1][w-1] = Out[h-2][w-2]

    Out = Out.clip(0)  
    Out = np.clip(Out, 0, 255).astype(np.uint8)   
    return Out    

def IntegralImage1(src):

    # + 1 to add padding - useful for pseudorecursion used in for-loop & SubIntegral Method
    rows = len(src) + 1
    cols = len(src[0]) + 1

    Out = np.zeros((rows,cols), np.uint32)
    for i in range(1, rows):
        for j in range(1, cols):
            Out[i][j] = src[i-1][j-1] + Out[i][j-1] + Out[i-1][j] - Out[i-1][j-1]

    return Out

def IntegralImage2(src):

    rows = len(src)
    cols = len(src[0])

    Out = np.zeros((rows,cols), np.uint32)
    Out = np.cumsum(src, axis = 0)
    Out = np.cumsum(Out, axis = 1)

    Out = np.concatenate((np.zeros((1,cols), np.uint32), Out), axis=0)
    Out = np.concatenate((np.zeros((rows+1,1), np.uint32), Out), axis=1)

    return Out

def SubIntegralImage(Int, root, dest):
    h, w = Int.shape

    # mandatory transformation because 
    # integral image is padded top & left-side
    root[0] = root[0] + 1
    root[1] = root[1] + 1
    dest[0] = dest[0] + 1
    dest[1] = dest[1] + 1
    # If rectangle boundaries exceed Image dimensions
    # Bring them to normal
    if(root[0] < 1):
        root[0] = 1
    if(root[1] < 1):
        root[1] = 1
    if(dest[0] >= h):
        dest[0] = h - 1
    if(dest[1] >= w):
        dest[1] = w - 1
    #add first - sub later to avoid ulong overflow
    return (Int[dest[0]][dest[1]] + Int[root[0]-1][root[1]-1] - Int[dest[0]][root[1]-1] - Int[root[0]-1][dest[1]])


NoiseFilenames = ["1_noise.png", "2_noise.png", "3_noise.png", "4_noise.png", "5_noise.png"]
OriginalFilenames = ["1_original.png", "2_original.png", "3_original.png", "4_original.png", "5_original.png"]
Image_Folder = "Original & Noisy/"

Original, Noisy, Filtered, Binary, Dilated = [], [], [], [], []

#change img_index to test all samples (0 - 4)
img_i = 0
MedianWindowSize = 3 
dilation_kernel_pars = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
dilation_kernel_words = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)

# read & scale
Original.append(cv.imread(Image_Folder+OriginalFilenames[img_i], cv.IMREAD_COLOR))
Original[0] = cv.resize(Original[0], None, fx=0.3, fy=0.3)

Noisy.append(cv.imread(Image_Folder+NoiseFilenames[img_i], cv.IMREAD_GRAYSCALE))
Noisy[0] = cv.resize(Noisy[0], None, fx=0.3, fy=0.3)

# filter for salt and pepper & smooth boundary noise &  binarise
Filtered.append(MedianFilter(Noisy[0], MedianWindowSize))
#Filtered[0] = cv.blur(Filtered[0], (3,3)) #if in use threshold 217 - else 209
Binary.append(cv.threshold(Filtered[0], 209, 255, cv.THRESH_BINARY_INV)[1]) 

# Segregate paragraphs by making them a whole component
Dilated.append(cv.dilate(Binary[0], dilation_kernel_pars, iterations = 5))  
# Extract info for each detected component
numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(Dilated[0])
# Compute Integral of Image
Integral = IntegralImage2(Filtered[0])
# Take Original Image to draw on it 
Output = cv.cvtColor(Original[0], cv.IMREAD_COLOR)

area, Pixel_Area, num_words = [], [], []
mean_grayscale = np.zeros(numLabels-1, np.float64)
# loop over the number of unique connected component labels
for i in range(1, numLabels):

	# extract the connected component statistics for the current label
    x = stats[i, cv.CC_STAT_LEFT]
    y = stats[i, cv.CC_STAT_TOP]   
    w = stats[i, cv.CC_STAT_WIDTH]	
    h = stats[i, cv.CC_STAT_HEIGHT]

    # area of the bounding box per Component
    area.append(stats[i, cv.CC_STAT_AREA])

    # area of letters (in pixels) per Component
    Pixel_Area.append(0)
    Component = np.zeros((h+1,w+1), np.uint8)
    for k in range(h):      
        for f in range(w):
            Component[k][f] = (Binary[0])[y+k][x+f] 
            if((Binary[0])[y+k][x+f] == 255):
                Pixel_Area[i-1] = Pixel_Area[i-1] + 1

    # grayscale mean value per Component
    mean_grayscale[i-1] = SubIntegralImage(Integral, [y,x], [y+h, x+w]) / (h * w)
    
    # number of words per Component
    Component = cv.dilate(Component, dilation_kernel_words, iterations = 3)
    num_words.append(cv.connectedComponentsWithStats(Component)[0])

	# a bounding box surrounding the connected Component   
    cv.rectangle(Output, (x,y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(Output, str(i), (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0 ,0, 255), 3)

    print("----Region " + str(i) + ":----\nArea (px): " + str(Pixel_Area[i-1]) + "\nBounding Box Area (px): " + str(area[i-1]) + "\nNumber of words: " + str(num_words[i-1]) + "\nMean gray-level value in bounding box: " + str(mean_grayscale[i-1]))


cv.imshow("lego", Output)
cv.waitKey(0)