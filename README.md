# BackgroundDetectionfromSurveillanceVideos

## install
### python2.7

## extract data file from video
- run transform/extract_frames.py
- check output: data/test.pkl

## separate background and foreground
- input: data/test.pkl
- run model/main.py
- check output: data/background.pkl

## annotate frames
- input[0]: data/background.pkl
- input[1]: data/test.pkl
- run model/BackgroundAnnotator.py
- TBD
