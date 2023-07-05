# **UWAT-GAN**
This code is the pytorch implementation of our paper "UWAT-GAN: Fundus Fluorescein Angiography Synthesis via Ultra-wide-angle Transformation Multi-scale GAN". It can be used to turning UWF scanning laser ophthalmoscopy(UWF-SLO) to the UWF fluorescein angiography(UWF-FFA) and display the tiny vascular lesion areas.  
<br><br>
![](/asset/Fig_1.png)
<br><br>
## UWF-SLO to UWF-FA at 3k resolution
![whole_result](/asset/whole_result.png)

## Pre-requisties
- Linux
- python>=3.7
- NVIDIA GPU (memory>=10G) + CUDA cuDNN
## Getting start to evaluate


### Install dependencies
```bash
pip install -r requirements.txt
```
### Configure the checkpoints
Fistly download the [checkpoint]() named as the **'UWFA-GAN_checkpoints'**, move it into the project directory and rename it to the **'checkpoints'**.
```
├── checkpoints
    ├──d_model_1_fine.pt
    ├──d_model_2_coarse
    ├──g_model_coarse
    ├──g_model_fine
``` 
### Evaluation dataset
Due the privacy of our dataset, we only provide 4 pictures for the result viewing. They are located at './example_pics/' 
```
├── example_pics
    ├──1.png
    ├──1-1.png
    ├──2.png
    ├──2-2.png
    ...
    ├──4-4.png
```
1.png means first UWF-SLO and the 1-1.png means first UWF-FA, 2 means the second pair, 3, 4, respectively.

### Evaluation
To do the evaluation process, run the following command:
```bash
python inference.py
```
After the evaluation, some new directories may be created.
the running results are saved in the directories './result_save' and two sub-directories called './Coarse_result' and './Fine_result'. 
```
├── result_save
    ├──Coarse_result
    ├──Fine_result
```
The './Coarse_result' saves the results 
coarse generator generates, while the './Fine_result' corresponds to results fine generator generates.  