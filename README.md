# Semantic segmentation of AR-Marker using SegNet
With this project I tried to segment AR markers, even if they are not detectable. The network is based on SegNet. The publication is [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561).

After a training on synthetical data, the network was able to provide an acceptable result on unseen synthetic data with a IoU of *79,56*.
But the result on real images is not very impressive, as the following examples shows.
![Result](/media/output.png)
**Even if the project does not meet the expectations, the implementation can be used for further projects.**
## How To Use It
There are two ways to train the net.
### Train On Synthetic Data
On the one hand with synthetic data. To do this, each second iteration is switched between the following two samples. Where one is generated completely synthetically and the other with a random background image from `./bg`.
![Data-Generator](/media/syn_gen.png)
For the above example the training was started as follows:
```
python3 train_synthetic_data.py -i 150000 -bs 4 -APRILTAG_36h11
```
A list with all arguments can be seen with the `-h` parameter.
### Train On Real Data
I have already tried to train SegNet on real data, but the generation of a enough large dataset was too time-consuming for me, so I don't have any results yet.
```
python3 train_real_data.py -i 150000 -bs 4 -dp ./train_dataset
```
A list with all arguments can be seen with the `-h` parameter.
### Do Prediction
You can run a prection with the following command: 
```
python3 predict.py -i ./train_dataset/0.jpg -o ./0_prediction.jpg
```
A list with all arguments can be seen with the `-h` parameter.
