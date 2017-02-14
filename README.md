# BackgroundDetectionfromSurveillanceVideos
Visit Wiki page for more details
- https://github.com/lingxiao1989/BackgroundDetectionfromSurveillanceVideos/wiki


## install
### python2.7

## sample video
- https://www.youtube.com/watch?v=7a3BSb281JM

## extract data file from video
- run transform/extract_frames.py
- check output: data/test.pkl

## [optional] apply external predictor
- run model/HaarPredictor.py to get mask

## converge on background
- input: data/test.pkl
- run model/main.py
- check output: data/background.pkl

## annotate frames
- input[0]: data/background.pkl
- input[1]: data/test.pkl
- run model/BackgroundAnnotator.py
- TBD
