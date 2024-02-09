# Computer-Vision with OpenCV
Projects in Python using Numpy, opencv-contrib-3.4.2.17 & Python 3.7 <br>
<h4>Project 1</h4>
In the first project we have to read images containing articles and processs <br>
in such way that we can extract information about them. We need to find <br>
the number of paragraphs, number of words per paragraph, the area of the <br>
bounding box per paragraph & the mean value of grayscale per paragraph.<br>
-> In order to complete those we have to filter the noisy images (salt & <br>
pepper noise), then binarise and use dilation to detect individual areas.<br>
For the grayscale mean value we calculate the subIntegral of the image after <br>
the calculation of the Integral Image. All filters and needed routines are <br>
implemented by the student (moi) except for the dilation. 

