# coldens_map
Python routine that creates total column density and rotational temperature maps from a series of moment 0 maps of the same molecule using a standard rotational diagram analysis (LTE analysis, optically thin emission) similar to what is used in synth_spect.py.

The routine reads the input.in main input file and an associated text file listing the names of the fits files. It computes the column density by reading online spectroscopic data through the JPL and CDMS databases. Two examples are given here: CH3OH transitions towards the Barnard 5 molecular clouds as observed with the IRAM 30m telescope and towards the NGC1333-IRAS4A low-mass protostar as observed with the Plateau de Bure interferometer. 


## input.in

The input parameters read by coldens_map.py are the following:

- species: Species to consider as specified in jpl or cdms database
- fileobs: Input file listing the fits files and the associated frequency
- prefix: Prefix of output files
- choice_y: Unit of intensity, "kelvin" for main-beam temperatures (moment 0 unit: K km/s) or "jansky" for fluxes (moment 0 unit: Jy km/s)
- choice_data: Where to look for spectro data (local or online, please specify online when first used to save data locally)
- database: Spectroscopic database (jpl or cdms)
- source_lon: Longitude for coordinates of reference source
- source_lat: Latitude for coordinates of reference source
- threshold: Threshold in steps of rms over which transitions are included in the rotational diagram


## fileobs

The python routine reads a second ASCII file that lists the frequency of each transition together with the name of the fits files of the moment 0 map, the associated rms noise of the map (in Jy km/s or K km/s depending on choice_y), and the beam size along the two axes (in arcsec). 