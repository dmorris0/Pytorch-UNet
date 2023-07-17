#!/usr/bin/env bash

days=('0723' '0724' '0725' '0726' '0727' '0728' '0729' '0730' '0731' '0801' '0802' '0803' '0804' '0805' '0806' '0807' '0808')
channels=('ch1' 'ch2' 'ch3' 'ch4' 'ch5' 'ch6' 'ch7' 'ch8')

for c in "${channels[@]}"; do
    for d in "${days[@]}"; do
        python track_eggs.py 54 --prefix "${c}_${d}" \
          --minseq 3 --minlen 10 --lost 15 --radius 50 \
          --videodir /mnt/research/3D_Vision_Lab/Hens/Hens_2021_sec \
          --imagedir /mnt/research/3D_Vision_Lab/Hens/ImagesJPG \
          --detectdir /mnt/scratch/f0106144/Hens_Detections_054 \
          --hideanno \
          --jsondir ../hen_srop/occluded_tracks/ \
          --vidtrackdir /mnt/scratch/f0106144/trackvid
    done
done

