#!/bin/bash
# for file in ~/Dropbox/horse_videos/Experimental_pain/Observer_not_present/No_pain/*.mts; do ffmpeg -i "$file" "${file%.mts}".jpg; done
# ffmpeg -ss 00:02:00 -i ~/Dropbox/horse_videos/Experimental_pain/Observer_not_present/No_pain/#1_1b.mts -vcodec mjpeg -t 00:00:10 -vf scale=320:240 -r 10 data_test/frame_%06d.jpg
# Below line just gave no grey frame!
ffmpeg -ss 00:02:11 -i data/Experimental_pain/Observer_not_present/No_pain/1_1b.mts -vcodec png -t 00:00:10 -vf scale=320:240 -r 10 -an data_test/frame_%06d.png
# Below jpg-line gives very pixeled images and poor quality.
# ffmpeg -ss 00:02:11 -i ~/Dropbox/horse_videos/Experimental_pain/Observer_not_present/No_pain/#1_1b.mts -vcodec mjpeg -t 00:00:10 -vf scale=320:240 -r 10 -an data_test/frame_%06d.jpg
# Below line gives 1-2Mb frames.
# ffmpeg -ss 00:02:11 -i ~/Dropbox/horse_videos/Experimental_pain/Observer_not_present/No_pain/#1_1b.mts -vcodec png -t 00:00:10 -r 10 -an data_test/frame_%06d.png
