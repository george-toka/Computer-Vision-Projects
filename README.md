# Computer-Vision with OpenCV
Projects in Python using Numpy, opencv-contrib-3.4.2.17 & Python 3.7 <br>
<h4>Project 1</h4>
In the first project we have to read images containing articles and processs <br>
in such way that we can extract information about them. We need to find <br>
the number of paragraphs, number of words per paragraph, the area of the <br>
bounding box per paragraph & the mean value of grayscale per paragraph.<br>
<br>
In order to complete those we have to filter the noisy images (salt & <br>
pepper noise), then binarise and use dilation to detect individual areas.<br>
For the grayscale mean value we calculate the subIntegral of the image after <br>
the calculation of the Integral Image. All filters and needed routines are <br>
implemented by the student (moi) except for the dilation. <br>

<h4>Project 2</h4>
<ins>Part 1</ins> : Here we have to perform image stitching between 4 images. We extract <br>
features using SIFT or SURF and we manually stitch them.<br>
<ins>Part 2</ins> : We perform calibration & undistortion to our mobile phone's camera <br>
and then  we estimate the depth of an object using pictures of two different angles.<br>

<h4>Project 3</h4>
<ins>Part 1</ins> : We perform image classification using machine learning algorithms. <br>
K-Means Clustering for the image feature classes. Train and test KNN and SVM models. <br>
<ins>Part 2</ins> : We perform image classification using convolutional networks. <br>
