#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"
set -e

# 1) Generate white background movie
cd light/
for file in untitled.*.ppm; 
do 
	convert $file -transparent white ${file:0:14}.png;
done
img2webp -o avatar-light.webp -q 30 -mixed -d 50 untitled*.png
cd ..

# 2) Generate black background movie
cd dark/
for file in untitled.*.ppm; 
do 
	convert $file -transparent black ${file:0:14}.png;
done
img2webp -o avatar-dark.webp -q 30 -mixed -d 50 untitled*.png
cd ..
