# hackbridge
Hackbridge data validation project

--

### steeldata.csv
File containing data about several hundred steels, that we will use as the initial dataset for this project.

Format is one row per material.  First column is an ID; then 13 columns of composition information (% of each element by weight; the remainder is iron);
then a column of heat treatment information; then three material properties (yield strength, ultimate tensile strength, elongation).  Note that not all 
the material properties have been measured for each material, represented by empty entries in the csv file.
