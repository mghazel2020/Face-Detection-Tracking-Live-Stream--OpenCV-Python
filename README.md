# Face Detection & Tracking from Live Video Stream using OpenCV-Python

<img src="images/banner/face-tracking.jpg" width="1000"/>

## 1. Objective 

The objective of this project is to demonstrate face detection and tracking from live stream video using built-in OpenCV Python Haar Cascade face detector and object tracking functionalities. 

## 2. Face Tracking Approach

* Our implemented approach is as follows: 

  * Once the camera starts capturing the video stream of the person's face in the field of view, we: 
    * Grab the first frame of the video stream and apply Haar Cascades face detector, in order to detect the person's face. 
    * The detected face, as localized by its bounding-box, represented our region of interest (ROI), which is then tracked in the subsequent frames, using the following moving object tracking algorithms:
      * Mean-Shift 
      * Cam-Shift 
      * BOOSTING 
      * MIL 
      * KCF 
      * TLD 
      * MEDIAN FLOW

 Next, we shall illustrate the implementation of this approach and illustrate the face tracking results for each object tracking approach. 

## 3. Development 

* Project: Face Tracking: 
* The objective of this project is to demonstrate face tracking from live stream video using seven built-in OpenCV Python tracking functionalities, as listed above. 
  * We shall assume the following: 
  * The moving camera is fixed or moving 
  * The face is moving 
  * We apply Haar Cascades to detect the face from the first frame of the live stream 
  * Our objective is to track the object of interest in the remaining frames. 

* Author: Mohsen Ghazel (mghazel) 
* Date: April 15th, 2021 

### 3.1. Step 1: Imports and global variables:

#### 3.1.1. Python imports:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>image <span style="color:#200080; font-weight:bold; ">as</span> mpimg

<span style="color:#595979; "># input/output OS</span>
<span style="color:#200080; font-weight:bold; ">import</span> os 

<span style="color:#595979; "># date-time to show date and time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime

<span style="color:#595979; "># Use %matplotlib notebook to get zoom-able &amp; resize-able notebook. </span>
<span style="color:#595979; "># - This is the best for quick tests where you need to work interactively.</span>
<span style="color:#595979; "># %matplotlib inline</span>

<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>


### 3.1.2. Global variables:

Setup and instantiate the Haar Cascades face detector:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Setup the folder comtaining the pre-trained Haar </span>
<span style="color:#595979; "># Cascades detector configuration XML files:</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
haar_cascades_configs_folder <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"C:/MyWebSites/MyWebsSite/MyProjects/Human-Face/resources/DATA/haarcascades/"</span>

<span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Instantiate the pre-trained Haar Cascades </span>
<span style="color:#595979; "># face detector using its configuration file</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Instantiate the pre-trained Haar Cascades face detector using its configuration file</span>
face_cascade <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>CascadeClassifier<span style="color:#308080; ">(</span>haar_cascades_configs_folder <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">'haarcascade_frontalface_default.xml'</span><span style="color:#308080; ">)</span>
</pre>


### 3.2. Step 2: Apply the Mean-Shift Tracker:

* In this section, we shall implement the Mean-Shift Tracker in OpenCV Python:
  * To track a single face from a live video stream


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># 2.1) open the video camera and capture a video stream</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot capture video stream!"</span><span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
<span style="color:#595979; "># take first frame of the video</span>
ret<span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Step 2.2) Detect the face from the first frame</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># decet the face from the frame</span>
face_rects <span style="color:#308080; ">=</span> face_cascade<span style="color:#308080; ">.</span>detectMultiScale<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">)</span> 

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Step 2.3) Setup the initial Tracking Window:</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># get the bounding-box of the detected face</span>
<span style="color:#308080; ">(</span>face_x<span style="color:#308080; ">,</span>face_y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> <span style="color:#400000; ">tuple</span><span style="color:#308080; ">(</span>face_rects<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> 
<span style="color:#595979; "># setup the tracking window</span>
track_window <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>face_x<span style="color:#308080; ">,</span>face_y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span>
<span style="color:#595979; "># set up the ROI tracking-window for tracking</span>
roi <span style="color:#308080; ">=</span> frame<span style="color:#308080; ">[</span>face_y<span style="color:#308080; ">:</span>face_y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">,</span> face_x<span style="color:#308080; ">:</span>face_x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">]</span>

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Setup 2.4: Start the Mean-Shift tracking of the setup </span>
<span style="color:#595979; ">#           initial window:</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Use the HSV Color Mapping</span>
hsv_roi <span style="color:#308080; ">=</span>  cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>roi<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Find histogram to backproject the target on each frame for calculation of meanshit</span>
roi_hist <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcHist<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv_roi<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#074726; ">None</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Normalize the histogram array values given a min of 0 and max of 255</span>
cv2<span style="color:#308080; ">.</span>normalize<span style="color:#308080; ">(</span>roi_hist<span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>NORM_MINMAX<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Setup the termination criteria, either 10 iteration or move by at least 1 pt</span>
term_crit <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_EPS <span style="color:#44aadd; ">|</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_COUNT<span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># iterate over the farmes</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    ret <span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> ret <span style="color:#44aadd; ">==</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
        
        <span style="color:#595979; "># Grab the Frame in HSV</span>
        hsv <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Calculate the Back Projection based off the roi_hist created earlier</span>
        dst <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcBackProject<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Apply meanshift to get the new coordinates of the rectangle</span>
        ret<span style="color:#308080; ">,</span> track_window <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>meanShift<span style="color:#308080; ">(</span>dst<span style="color:#308080; ">,</span> track_window<span style="color:#308080; ">,</span> term_crit<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># Display the tracking results:</span>
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># Draw the new rectangle on the image</span>
        x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h <span style="color:#308080; ">=</span> track_window
        img2 <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">,</span>y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># display the image</span>
        cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Face Tracker: Mean-Shift'</span><span style="color:#308080; ">,</span>img2<span style="color:#308080; ">)</span>
    
        <span style="color:#595979; "># increment the frame counter</span>
        frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
        
        <span style="color:#595979; "># quit if user hits: ESC</span>
        k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
        <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span>
            <span style="color:#200080; font-weight:bold; ">break</span>
        
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">break</span>

<span style="color:#595979; "># close all windows</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># realease the camera resources</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

* Sample facing tracking results using the Mean-Shift tracker are illustrated next.

<table>
  <tr>
    <td> <img src="images/Mean-Shift-01/Capture-01.PNG"  width="500" ></td>
    <td> <img src="images/Mean-Shift-01/Capture-02.PNG"  width="500" ></td>
   </tr> 
   <tr>
    <td> <img src="images/Mean-Shift-01/Capture-03.PNG"  width="500" ></td>
    <td> <img src="images/Mean-Shift-01/Capture-04.PNG"  width="500" ></td>
  </td>
  </tr>
</table>


### 3.3. Step 3: Cam-Shift Tracker:

* In this section, we shall implement the Cam-Shift Tracker in OpenCV Python:
  * To track a single face from a live video stream
  
  
<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># 3.1) open the video camera and capture a video stream</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot capture video stream!"</span><span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
<span style="color:#595979; "># take first frame of the video</span>
ret<span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Step 3.2) Detect the face from the first frame</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># decet the face from the frame</span>
face_rects <span style="color:#308080; ">=</span> face_cascade<span style="color:#308080; ">.</span>detectMultiScale<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">)</span> 

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Step 3.3) Setup the initial Tracking Window:</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># get the bounding-box of the detected face</span>
<span style="color:#308080; ">(</span>face_x<span style="color:#308080; ">,</span>face_y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> <span style="color:#400000; ">tuple</span><span style="color:#308080; ">(</span>face_rects<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> 
<span style="color:#595979; "># setup the tracking window</span>
track_window <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>face_x<span style="color:#308080; ">,</span>face_y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span>
<span style="color:#595979; "># set up the ROI tracking-window for tracking</span>
roi <span style="color:#308080; ">=</span> frame<span style="color:#308080; ">[</span>face_y<span style="color:#308080; ">:</span>face_y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">,</span> face_x<span style="color:#308080; ">:</span>face_x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">]</span>

<span style="color:#595979; "># Use the HSV Color Mapping</span>
hsv_roi <span style="color:#308080; ">=</span>  cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>roi<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Find histogram to backproject the target on each frame for calculation of meanshit</span>
roi_hist <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcHist<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv_roi<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#074726; ">None</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Normalize the histogram array values given a min of 0 and max of 255</span>
cv2<span style="color:#308080; ">.</span>normalize<span style="color:#308080; ">(</span>roi_hist<span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>NORM_MINMAX<span style="color:#308080; ">)</span>

<span style="color:#595979; "># Setup the termination criteria, either 10 iteration or move by at least 1 pt</span>
term_crit <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_EPS <span style="color:#44aadd; ">|</span> cv2<span style="color:#308080; ">.</span>TERM_CRITERIA_COUNT<span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># iterate over the farmes</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    ret <span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> ret <span style="color:#44aadd; ">==</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
        
        <span style="color:#595979; "># Grab the Frame in HSV</span>
        hsv <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2HSV<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Calculate the Back Projection based off the roi_hist created earlier</span>
        dst <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>calcBackProject<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>hsv<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span>roi_hist<span style="color:#308080; ">,</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">180</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Apply Camshift to get the new coordinates of the rectangle</span>
        ret<span style="color:#308080; ">,</span> track_window <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>CamShift<span style="color:#308080; ">(</span>dst<span style="color:#308080; ">,</span> track_window<span style="color:#308080; ">,</span> term_crit<span style="color:#308080; ">)</span>
       
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># Display the tracking results:</span>
        <span style="color:#595979; ">#----------------------------------</span>
        <span style="color:#595979; "># Draw the new rectangle on the image</span>
        pts <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>boxPoints<span style="color:#308080; ">(</span>ret<span style="color:#308080; ">)</span>
        pts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>int0<span style="color:#308080; ">(</span>pts<span style="color:#308080; ">)</span>
        img2 <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>polylines<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span><span style="color:#308080; ">[</span>pts<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#074726; ">True</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># display the image</span>
        cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Face Tracker: Cam-Shift'</span><span style="color:#308080; ">,</span>img2<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># increment the frame counter</span>
        frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
        
        <span style="color:#595979; "># quit if user hits: ESC</span>
        k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
        <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span><span style="color:#308080; ">:</span>
            <span style="color:#200080; font-weight:bold; ">break</span>
        
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">break</span>
        
<span style="color:#595979; "># close all windows</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># realease the camera resources</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

* Sample facing tracking results using the Cam-Shift tracker are illustrated next.

<table>
  <tr>
    <td> <img src="images/Cam-Shift-02/Capture-01.PNG"  width="500" ></td>
    <td> <img src="images/Cam-Shift-02/Capture-02.PNG"  width="500" ></td>
   </tr> 
   <tr>
    <td> <img src="images/Cam-Shift-02/Capture-03.PNG"  width="500" ></td>
    <td> <img src="images/Cam-Shift-02/Capture-04.PNG"  width="500" ></td>
  </td>
  </tr>
</table>

### 3.4. Step 4: OpenCV Object Tracking API:

* In this section, we shall implement the Tracking APIs (Built-in with OpenCV):
  * We get the following options to experiment with the following trackers:
    * Enter 0 for BOOSTING
    * Enter 1 for MIL
    * Enter 2 for KCF
    * Enter 3 for TLD
    * Enter 4 for MEDIANFLOW
    

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">'''</span>
<span style="color:#595979; ">Gets the user tracker selection:</span>
<span style="color:#595979; ">'''</span>
<span style="color:#200080; font-weight:bold; ">def</span> ask_for_tracker<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Please select the Tracker API would you like to use:"</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 0 for BOOSTING: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 1 for MIL: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 2 for KCF: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 3 for TLD: "</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Enter 4 for MEDIANFLOW: "</span><span style="color:#308080; ">)</span>
    choice <span style="color:#308080; ">=</span> <span style="color:#400000; ">input</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Please select your tracker: "</span><span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'0'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerBoosting_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'1'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerMIL_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'2'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerKCF_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'3'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerTLD_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">if</span> choice <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'4'</span><span style="color:#308080; ">:</span>
        tracker <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>TrackerMedianFlow_create<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>


    <span style="color:#200080; font-weight:bold; ">return</span> tracker
</pre>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Get the tracker user option</span>
tracker <span style="color:#308080; ">=</span> ask_for_tracker<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># Get the tracker user option name</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"User selected Tracker: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>tracker<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>split<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>

Please select the Tracker API would you like to use:
Enter <span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">for</span> BOOSTING<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">1</span> <span style="color:#200080; font-weight:bold; ">for</span> MIL<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">2</span> <span style="color:#200080; font-weight:bold; ">for</span> KCF<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">3</span> <span style="color:#200080; font-weight:bold; ">for</span> TLD<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">4</span> <span style="color:#200080; font-weight:bold; ">for</span> MEDIANFLOW<span style="color:#308080; ">:</span> 
Please select your tracker<span style="color:#308080; ">:</span> <span style="color:#008c00; ">1</span>
User selected Tracker<span style="color:#308080; ">:</span> TrackerMIL
</pre>



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Get the user Tracker-option:</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># Get the tracker option from the user</span>
tracker <span style="color:#308080; ">=</span> ask_for_tracker<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># get the tracker name</span>
tracker_name <span style="color:#308080; ">=</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>tracker<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>split<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span>

<span style="color:#595979; ">#------------------------------------------------------------</span>
<span style="color:#595979; "># open the video camera and capture a video stream</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
cap <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>VideoCapture<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># check the status of the opened video file</span>
<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#200080; font-weight:bold; ">not</span> cap<span style="color:#308080; ">.</span>isOpened<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cannot capture video stream!"</span><span style="color:#308080; ">)</span>
    exit<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>
    
<span style="color:#595979; "># take first frame of the video</span>
ret<span style="color:#308080; ">,</span>frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Special function allows us to draw on the very first frame our desired ROI</span>
roi <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>selectROI<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> <span style="color:#074726; ">False</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># Initialize tracker with first frame and bounding box</span>
ret <span style="color:#308080; ">=</span> tracker<span style="color:#308080; ">.</span>init<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> roi<span style="color:#308080; ">)</span>

<span style="color:#595979; "># frame counter</span>
frame_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">;</span>

<span style="color:#595979; "># iterate over the farmes</span>
<span style="color:#200080; font-weight:bold; ">while</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; "># Read a new frame</span>
    ret<span style="color:#308080; ">,</span> frame <span style="color:#308080; ">=</span> cap<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># Update tracker</span>
    success<span style="color:#308080; ">,</span> roi <span style="color:#308080; ">=</span> tracker<span style="color:#308080; ">.</span>update<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># roi variable is a tuple of 4 floats</span>
    <span style="color:#595979; "># We need each value and we need them as integers</span>
    <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>y<span style="color:#308080; ">,</span>w<span style="color:#308080; ">,</span>h<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> <span style="color:#400000; ">tuple</span><span style="color:#308080; ">(</span><span style="color:#400000; ">map</span><span style="color:#308080; ">(</span><span style="color:#400000; ">int</span><span style="color:#308080; ">,</span>roi<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Draw Rectangle as Tracker moves</span>
    <span style="color:#200080; font-weight:bold; ">if</span> success<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># Tracking success</span>
        p1 <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span> y<span style="color:#308080; ">)</span>
        p2 <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>x<span style="color:#44aadd; ">+</span>w<span style="color:#308080; ">,</span> y<span style="color:#44aadd; ">+</span>h<span style="color:#308080; ">)</span>
        cv2<span style="color:#308080; ">.</span>rectangle<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> p1<span style="color:#308080; ">,</span> p2<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">else</span> <span style="color:#308080; ">:</span>
        <span style="color:#595979; "># Tracking failure</span>
        cv2<span style="color:#308080; ">.</span>putText<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">"Failure to Detect Tracking!!"</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">100</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">200</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>FONT_HERSHEY_SIMPLEX<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># Display tracker type on frame</span>
    cv2<span style="color:#308080; ">.</span>putText<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> tracker_name<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">20</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">400</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>FONT_HERSHEY_SIMPLEX<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">255</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span><span style="color:#308080; ">;</span>

    <span style="color:#595979; ">#------------------------------------------------------------</span>
    <span style="color:#595979; "># Display the tracking results:</span>
    <span style="color:#595979; ">#------------------------------------------------------------</span>
    <span style="color:#595979; "># Draw the new rectangle on the image</span>
    cv2<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Face Tracker: '</span> <span style="color:#44aadd; ">+</span> tracker_name<span style="color:#308080; ">,</span> frame<span style="color:#308080; ">)</span>
       
    <span style="color:#595979; "># increment the frame counter</span>
    frame_counter <span style="color:#308080; ">=</span> frame_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
        
    <span style="color:#595979; "># Exit if ESC pressed</span>
    k <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>waitKey<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">&amp;</span> <span style="color:#008c00; ">0xff</span>
    <span style="color:#200080; font-weight:bold; ">if</span> k <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">27</span> <span style="color:#308080; ">:</span> 
        <span style="color:#200080; font-weight:bold; ">break</span>
<span style="color:#595979; ">#------------------------------------------------------------        </span>
<span style="color:#595979; "># close all windows</span>
<span style="color:#595979; ">#------------------------------------------------------------</span>
cv2<span style="color:#308080; ">.</span>destroyAllWindows<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># realease the camera resources</span>
cap<span style="color:#308080; ">.</span>release<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>


Please select the Tracker API would you like to use:
Enter <span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">for</span> BOOSTING<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">1</span> <span style="color:#200080; font-weight:bold; ">for</span> MIL<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">2</span> <span style="color:#200080; font-weight:bold; ">for</span> KCF<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">3</span> <span style="color:#200080; font-weight:bold; ">for</span> TLD<span style="color:#308080; ">:</span> 
Enter <span style="color:#008c00; ">4</span> <span style="color:#200080; font-weight:bold; ">for</span> MEDIANFLOW<span style="color:#308080; ">:</span> 
Please select your tracker<span style="color:#308080; ">:</span> <span style="color:#008c00; ">1</span>
</pre>

* Sample facing tracking results using the BOOSTING Tracker are illustrated next.. Similar tracking results can be obtained by selecting and executing the other four trackers. 


<table>
  <tr>
    <td> <img src="images/Boosting/Capture-00.PNG"  width="500" ></td>
    <td> <img src="images/Boosting/Capture-01.PNG"  width="500" ></td>
   </tr> 
   <tr>
    <td> <img src="images/Boosting/Capture-02.PNG"  width="500" ></td>
    <td> <img src="images/Boosting/Capture-03.PNG"  width="500" ></td>
  </td>
  <tr>
    <td> <img src="images/Boosting/Capture-04.PNG"  width="500" ></td>
    <td> <img src="images/Boosting/Capture-05.PNG"  width="500" ></td>
  </td>
  </tr>
</table>

### 3.5. Step 5: Display a successful execution message:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">15</span> <span style="color:#008c00; ">18</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">8</span><span style="color:#308080; ">:</span><span style="color:#008000; ">21.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>


## 4. Analysis

* We have demonstrated single face detection and tracking from live video stream using 7 tracking algorithms implemented in OpenCV Python:

  * The 5 tracking algorithms implemented in the OpenCV Tracking API all performed equally and extremely well, yielding nearly perfect tracking of the moving object of interest.
  * BOOSTING
  * MIL
  * KCF
  * TLD
  * MEDIAN FLOW
  * The other 2 tracking algorithms generally yield inadequate face tracking results, especially when the face is turned sideway form the camera:
    * Mean-Shift
    * Cam-Shift

## 5. Future Work

* We propose to explore the following related issues:

  * To explore these implemented tracking algorithm and get a better understating of:
    * How each algorithm works
    * The advantages and limitations of each algorithm
    * To implement multi-object trackers for tracking and distinguishing multiple objects at the same time.

## 6. References

1. Adrian Rosebrock. OpenCV Object Tracking. https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/ 
2. Adrian Rosebrock. Simple object tracking with OpenCV. https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/ 
3. Satya Mallick. Object Tracking using OpenCV. https://learnopencv.com/object-tracking-using-opencv-cpp-python/ 
4. Anna Petrovicheva. Multiple Object Tracking in Realtime. https://opencv.org/multiple-object-tracking-in-realtime/ 
5. Ehsan Gazar. Object Tracking with OpenCV. https://ehsangazar.com/object-tracking-with-opencv-fd18ccdd7369 
6. Automatic Addison. Real-Time Object Tracking Using OpenCV and a Webcam. https://automaticaddison.com/real-time-object-tracking-using-opencv-and-a-webcam/ 
7. Automatic Addison. How to Do Multiple Object Tracking Using OpenCV. https://automaticaddison.com/how-to-do-multiple-object-tracking-using-opencv/




  
  
