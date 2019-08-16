# Genre Classification
Building a predictive model for classification of audio files into musical genres.  
http://jakeallensmith.com/projects/genre-classification/

## Objectives
- Produce a predictive model for classifying audio files into musical genres.

## Dataset
- A collection of waveform descriptors calculated on my personal music library.

## Contents
- calculate_descriptors.Rmd  
Calculation of waveform descriptors using functions in the *tuneR* and *seewave* packages.

- final_music_descriptors.csv  
The dataset used in the final analysis.

- genre_classifier.Rmd  
Exploratory analysis, cleaning, and transformation of dataset. Tuning and testing of predictive models.

- less_features.Rmd  
A multinomial regression model with elasticnet penalties using a truncated dataset.

- music_descriptors_short.csv  
The initially calculated waveform descriptors.

- music_library.csv  
A table of metadata and file paths pulled from my music collection.

- writeup.Rmd  
A final report.
