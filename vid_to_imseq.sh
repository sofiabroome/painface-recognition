#!/bin/bash
# for file in ~/Dropbox/horse_videos/Experimental_pain/Observer_not_present/No_pain/*.mts; do ffmpeg -i "$file" "${file%.mts}".jpg; done
ffmpeg -i ~/Dropbox/horse_videos/20120407140353.mts -vf scale=320:-1 -r 1 data/output_%04d.jpg

