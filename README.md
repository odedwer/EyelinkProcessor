# EyeLinkProcessor
Eyetracking processor for Eyelink eyetrackers with syncing to mne raw eeg timeseries objects.

In this repository you will find several classes designed for Eyetracking data processing and preliminary analysis.

## Design
The main class is EyeLinkProcessor, which saves all the data read from the .asc file.
This class uses a given parser object, inheriting from BaseETParser to read the .asc file into the different fields, which are converted to Pandas dataframes at the end of parsing.
A saccade detector can also be specified, if you want to use an external saccade detection algorithm, like Enbgert & Morgenthaler for microsaccade detection.

## Expanding and using the repository
To expand the repository to other types of data/saccade detectors, one needs to implement a class inheriting from one of base classes and add that class to the corresponding Enum class in Enums.py
you can find implementation examples in the EyeTrackingParsers and SaccadeDetectors files.

## Disclaimer
The code in this repository has undergone some testing, but bugs might still happen. 
Please use with caution, I suggest comparing to known results from other sources before depending upon this repository completely.


