#!/bin/bash
# for file in ~/Dropbox/horse_videos/Experimental_pain/Observer_not_present/No_pain/*.mts; do ffmpeg -i "$file" "${file%.mts}".jpg; done
ffmpeg -i ~/Dropbox/horse_videos/Experimental_pain/Observer_not_present/No_pain/#1_1b.mts -vf scale=320:-1 -r 1 data/output_%04d.jpg

