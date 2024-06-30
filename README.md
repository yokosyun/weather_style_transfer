# Neural Style Transfer for weather editing


## Install
```
python3.8 -m venv ~/venv/nst
source ~/venv/nst/bin/activate
pip install pip --upgrade
pip install -r requirements.txt
pre-commit install
```

## Run NST
```
python neural_style_transfer.py --content_img_name lion.jpg --style_img_name vg_starry_night.jpg
python neural_style_transfer.py --content_img_name berlin_000000_000019_leftImg8bit.png --style_img_name rain_storm-001.jpg
```


## Reconstruct image from representation
```
python3 reconstruct_image_from_representation.py --should_reconstruct_content True --should_visualize_representation False
```

## Debugging/Experimenting

Q: L-BFGS can't run on my computer it takes too much GPU VRAM?<br/>
A: Set Adam as your default and take a look at the code for initial style/content/tv weights you should use as a start point.

Q: Output image looks too much like style image?<br/>
A: Decrease style weight or take a look at the table of weights (in neural_style_transfer.py), which I've included, that works.

Q: There is too much noise (image is not smooth)?<br/>
A: Increase total variation (tv) weight (usually by multiples of 10, again the table is your friend here or just experiment yourself).


## Reference
[this accompanying YouTube video](https://www.youtube.com/watch?v=XWMwdkaLFsI)
PyTorch implementation of the original NST paper (:link: [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)).
