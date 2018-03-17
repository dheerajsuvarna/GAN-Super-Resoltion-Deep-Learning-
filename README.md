## Superresolution_With_GANs

### Using CelebA dataset
- Download the Align&Cropped Images from the CelebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- Unzip all pictures into any folder which is within a folder i.e. the folder structure should have the form folder1/folder2/<unzipped images>. Make sure no other folder exists whithin folder1
- run the train script using command
```bash
./train --dataset=folder --dataroot="<folder1_location>" --imageSize=89 --cuda
```
