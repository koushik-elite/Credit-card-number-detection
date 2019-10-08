# Credit card number detection
Detect Credit card number using Mask RCNN and make task easier for OCR to retrieve number from the card

The idea is to change traditional card based transaction using computer vision, Consider if we have a credit card device uses camera (computer vision) instead of swiping, So there is no need of physical card, even we can reduce the usage of plastic cards. 

## Preparing the dataset

I downloaded some credit card images with variable size from https://www.moneyhero.com.hk/api/credit-card/v2/cards/all?lang=en&pageSize=1000 REST Service 1000 samples for train and test

### Demo ScreenShot 1
![ScreenShot_1](images/image_6.jpg?raw=true "ScreenShot_1")

Then i used free Image Polygonal Annotation tool for creating annotation over the image like create a box marker around the card number on credit card image.

### Demo ScreenShot 2
Tool [LabelMe](https://github.com/wkentaro/labelme)

![ScreenShot_3](images/labelme2.png?raw=true "ScreenShot_3")
![ScreenShot_2](images/labelme1.png?raw=true "ScreenShot_2")


finally run [labelme2coco.py](labelme2coco.py) file to convert all the Image annotation json to COCO like dataset with mask image


## 1. Installation

Download Anaconda

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.anaconda.com/archive/Anaconda3-2018.12-Windows-x86_64.exe
[win32]: https://repo.anaconda.com/archive/Anaconda3-2018.12-Windows-x86.exe
[mac64]: https://repo.anaconda.com/archive/Anaconda3-2018.12-MacOSX-x86_64.sh
[lin64]: https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
[lin32]: https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86.sh

**Install** [Anaconda](https://docs.anaconda.com/anaconda/install/) on your machine. Detailed instructions:

## 2. Create and Activate the Environment

Please go though this [doc](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) before you creating an environment.
After that create a environment using following command

```
conda create --name deep-learning
```

Then activate the environment using following command

```
activate deep-learning
```

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

**Now, you can create a local version of the project**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/koushik-elite/Face-Generation.git
cd TV-Script-Generation
```

2. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch -c pytorch
	pip install torchvision
	```

3. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

## Training

Creating image dataset and training process is available in [training-code.py](training-code.py) file
```
python training-code.py
```

## Prediction

predict mask image for single credit card image is available in [predict.py](predict.py) file
```
python predict.py
```