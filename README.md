# low-redshift-survey

## PROSP.py
- builds my obs dictionary

## mock
- builds the fake thetas and build the mock data as the dis_data file

## ResEscFrac
- reads the outputs and puts them in a iot dictionary, then puts the escape fractions into a nice plot

## Plot.py
- contains all the functions I use for plotting everything like Corner, subcorner_custom, Res, Plot_Phot & Plot_Spec

## Plot.ipynb
- uses all the functions contained in Plot.py

## Build.py
- contains the build_obs, build_obs_dis, build_model, build_sps and very importantly build_output, used by ResEscFrac.ipynb

## toolbox.py
- tool kit with all the functions I need to analyze the output of prospector
