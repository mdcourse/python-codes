#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"
set -e

# 1) Generate white background movie
for file in _light.*.ppm; 
do 
	convert -resize 50% $file -transparent white ${file:0:12}.png;
done
img2webp -o avatar.webp -q 30 -mixed -d 25 _light*.png

# 2) Generate black background movie
for file in _dark.*.ppm; 
do 
	convert -resize 50% $file -transparent black ${file:0:11}.png;
done
img2webp -o avatar-dm.webp -q 30 -mixed -d 25 _dark*.png
