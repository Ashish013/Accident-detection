# Accident-detection
Detecting accidents/crashes in CCTV footage.

This repository contains 3D Residual Network code for my minor project coursework titled 'Accident Identification using Deep Learning'. Please find the detailed report along side this repository [here](/assests/report.pdf).

### 3D ResNet details
Let x denote the input clip of size 3×L×H×W, where L is the number of frames inthe input video, H and W are the video frame height and width, and 3 refers to the RGB channels. In this model we consider each block to consist of two convolutional layers with a ReLU activation function after each layer, without the bottleneck layers.  Let z<sub>i</sub> ,a<sub>i</sub> be the tensors computed by the i<sup>th</sup> convolutional block in the residual network and  the activations obtained after applying ReLu function respectively The output of this i<sup>th</sup> residual block can be represented as:

a<sup>[i+2]</sup> = g<sup>[i]</sup> (a<sup>[i]</sup> + z<sup>[i+2]</sup>)

The  tensor z<sub>i</sub> in  this  case  is  4D  and  has  size N<sub>i</sub>×L×H<sub>i</sub>×W<sub>i</sub>,  where N<sub>i</sub> is  thenumber of filters used in the i<sup>th</sup> block. Each filter is 4-dimensional and it has size N<sub>i</sub>×t×d×d where t denotes the temporal extent of the filter and d denotes thespatial extent of the filter. The filters are then convolved in 3 dimensions i.e., overboth time and space dimensions.  The outputs from these convloutional layers are aggregated to the bottom layer where global average pooling takes place over the entire spatio-temporal volume and the final classification prediction is addressed by a fully connected layer as seen in the figure below:

|<img src=/assets/3D_Resnet.png/ width=150 height=300>| 
|:--:| 
| *3D ResNet Architecture.* |

### Dataset used
The model was trained using 615 video snippets scraped from the internet, with clip lengths ranging from 6 to 100 seconds. A subset of the original data-set is available for download [here](https://drive.google.com/uc?export=download&confirm=z7yV&id=1O5uVsSYa3zNS2Y344SkSWGniWWmQ4nFY). To train the model, download the data set and place it in the root directory.

### Running the code
- Download the pretrained weight from [here](https://drive.google.com/uc?export=download&confirm=cplh&id=1JfXAa0hR1oB9LGIz0SBO8RVbKz0jP5Tq) and place it in the directory of the notebook.
- Run the model in inference mode from the notebook.

### Hyperparameter values used
Hyperparameter | Value used
---------------|------------
Batch size | 16
Batch accumlation | 2
Sampled frames| 40 (20x2)
Learning rate | 0.006
Epochs | 50
No. of GPU's | 2

### Results

The following table summarises the validation results of the 3D ConvNets:

Model Architecture | Accuracy | F-score
----------|-----------|----------------
R2plus1d_18 | 87.37 | 0.869
Mc3_18 | 85.35 | 0.856
R3d_18 | 84.11 | 0.847

### To-do-list
- [ ] Deploy an app for easier access.
- [ ] Convert the notebook code into modular scripts.
