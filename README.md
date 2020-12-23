# fishdentification

## Goal

To classify fish by species as described in [Automatic fish classification for underwater species behaviour understanding](https://www.researchgate.net/publication/228985431_Automatic_fish_classification_for_underwater_species_behavior_understanding) by Spampinato et. al.

## Scope

Below is a diagram of the work done in the original paper. The red box indicates what is covered in this repo.

<p align="center">
  <img src="/fishdentification.jpg">
</p>

## Running the project

1. Install the Python packages used by the program: `pip install -r requirements.txt`. it is recommended that you install these into a Python virtual environment, to not interfere with any other versions of these packages that you may have installed for other projects.

2. Download fish pictures and their respective masks from the Fish4Knowledge dataset: http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/

   If you use the project as-is, you will need at least 300 photos and masks, thus fish numbered `01-05` and `07` are best.

3. Unzip the fish and masks in the `data` folder of this project, so that fish and masks are all arranged in a flat list, as such:

<p align="center">
  <img src="/readme_flatlist.jpg">
</p>

4. **run gofish.py**. The program will train and test itself for the number of times hardcoded in gofish.py (3, by default.), then print the results.
