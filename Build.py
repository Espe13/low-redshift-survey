
import sys
import os
import numpy as np
import pandas as pd
import pickle
import math
from astropy.io import fits
from sedpy.observate import load_filters
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.models.sedmodel import SpecModel, LineSpecModel, PolySpecModel
from prospect.io import write_results as writer
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
from prospect.sources.galaxy_basis import CSPSpecBasis
import matplotlib.pyplot as plt
from prospect.utils.obsutils import fix_obs
import prospect.io.read_results as reader


def build_obs(objid=1, el_table="lzlcs_optlines_obs.csv", phot_table='GP_Aperture_Matched_Photometry_v0.fits', err_floor=0.05, err_floor_el=0.05, **kwargs):
    

    num_to_id_translate = {'3307':'J003601+003307',
                    '15440':'J004743+015440',
                    '223':'J011309+000223',
                    '52044':'J012217+052044',
                    '145935':'J012910+145935',
                    '414608':'J072326+414608',
                    '472607':'J080425+472607',
                    '414146':'J081112+414146',
                    '211459':'J081409+211459',
                    '182052':'J082652+182052',
                    '480541':'J083440+480541',
                    '392925':'J090918+392925',
                    '183108':'J091113+183108',
                    '523960':'J091207+523960',
                    '505009':'J091208+505009',
                    '315221':'J091703+315221',
                    '395714':'J092552+395714',
                    '510925':'J093355+510925',
                    '593244':'J094001+593244',
                    '405249':'J095236+405249',
                    '235709':'J095700+235709',
                    '202508':'J095838+202508',
                    '523251':'J101401+523251',
                    '633308':'J102615+633308',
                    '635317':'J103344+635317',
                    '452718':'J103816+452718',
                    '474357':'J105117+474357',
                    '523753':'J105331+523753',
                    '485724':'J105525+485724',
                    '475204':'J110452+475204',
                    '412052':'J112224+412052',
                    '524509':'J112848+524509',
                    '493525':'J112933+493525',
                    '651341':'J113304+651341',
                    '312559':'J115855+312559',
                    '382422':'J115959+382422',
                    '305326':'J120934+305326',
                    '453930':'J121915+453930',
                    '63556':'J123519+063556',
                    '212716':'J124033+212716',
                    '21540':'J124423+021540',
                    '444902':'J124619+444902',
                    '123403':'J124835+123403',
                    '464535':'J124911+464535',
                    '255609':'J125503+255609',
                    '410221':'J125718+410221',
                    '451057':'J130059+451057',
                    '510451':'J130128+510451',
                    '422638':'J130559+422638',
                    '214817':'J131037+214817',
                    '104739':'J131419+104739',
                    '510309':'J131904+510309',
                    '421824':'J132633+421824',
                    '573315':'J132937+573315',
                    '112848':'J134559+112848',
                    '662438':'J135008+662438',
                    '612115':'J140333+612115',
                    '434435':'J141013+434435',
                    '461937':'J144010+461937',
                    '370512':'J151707+370512',
                    '572442':'J154050+572442',
                    '403325':'J155945+403325',
                    '81959':'J160437+081959',
                    '313054':'J164607+313054',
                    '495751':'J164849+495751',
                    '542133':'J172010+542133'}
    id_to_num_translate = {v: k for k, v in num_to_id_translate.items()}
    translate_el={'O2_3726A':'[OII]3726',
    'O2_3729A':'[OII]3729',
    'Ne3_3869A':'[NeIII]3870',
    'H1r_3889A':'HeI 3889',
    'Ne3_3967A':'[NeIII]3968',
    'H1r_3970A':'H 3970',
    'H1r_4102A':'H delta 4102',
    'H1r_4341A':'H gamma 4340',
    'O3_4363A':'[OIII]4364',
    'He2r_4686A':None,
    'H1r_4861A':'H beta 4861',
    'O3_4959A':'[OIII]4960',
    'O3_5007A':'[OIII]5007',
    'He1r_5876A':'HeI 5877',
    'O1_6300A':'[OI]6302',
    'N2_6548A':'[NII]6549',
    'H1r_6563A':'H alpha 6563',
    'N2_6584A':'[NII]6585',
    'S2_6716A':'[SII]6717',
    'S2_6731A':'[SII]6732'}

    def enumerate_strings(strings):
        enumerated_dict = {string: str(index + 1) for index, string in enumerate(strings)}
        return enumerated_dict
    translate_name = enumerate_strings(id_to_num_translate.keys())
    translate_name_rev = {v: k for k, v in translate_name.items()}

    path_wdir = '/Users/amanda/Desktop/Paper/technical/'
    path_data = os.path.join(path_wdir, 'data/')
    path_output = os.path.join(path_wdir, 'prospector/')
    path_flury = os.path.join(path_wdir, 'data/flury/')
    filternames = ['sdss_u0','sdss_g0','sdss_r0','sdss_i0','sdss_z0','galex_FUV','galex_NUV']
    EL_info = pd.read_csv('/Users/amanda/opt/anaconda3/envs/astro/lib/python3.10/site-packages/fsps/data/emlines_info.dat', header=None, sep = ',')

    # choose galaxy and read in the data
    phot_cat = fits.open(os.path.join(path_flury, phot_table))
    phot = phot_cat[1].data

    el = Table.read(os.path.join(path_flury, el_table), format="ascii")
    el.add_column([i for i in range(1,67)], name='id', index=0) #uniform reference for name now possible

    #two different inputs are possible for the objid, so we have to combine them into one ID we can work with:
    if objid>70:
        idx_gal = translate_name[num_to_id_translate[str(objid)]]
    elif objid==0:
        print('ERROR = objid can not be zero, the enumerations starts at 1')
    else:
        idx_gal = int(phot[objid-1][0])

    #id will be used to access the photometry, here everything starts from 0

    id = idx_gal-1

    #control line gets printed:
    print('GALAXIE: '+translate_name_rev[str(idx_gal)]+', ID: ' + str(idx_gal) + ', PHOTOMETRY: ' + phot_table + ', EMISSION_LINES: '+ el_table)

    #create all the lists I need for the obs dictionary:

#-------------------------------PHOTOMETRY-------------------------------------
    maggies = []
    maggies_unc = []
    phot_mask = []
    filters = []

    #MAGGIES AND MAGGIES_UNC:

    fil = ['FUV', 'NUV', 'U', 'G', 'R', 'I', 'Z']
    for x in fil:
        if phot['probPSF'][id]==1:
            if phot['psf_mag_'+x][id] > 0:
                m = 10**((phot['psf_mag_'+x][id]-8.9)/(-2.5))
                m_err = np.abs(m - 10**(((phot['psf_mag_'+x][id]+phot['psf_magerr_'+x][id])-8.9)/(-2.5)))
                maggies.append(m/3631)
                maggies_unc.append(m_err/3631)
            else:
                maggies.append(None)
                maggies_unc.append(None)
        else:
            if phot['aper_mag_3p1_'+x][id] > 0:
                m = 10**((phot['aper_mag_3p1_'+x][id]-8.9)/(-2.5))
                m_err = np.abs(m - 10**(((phot['aper_mag_3p1_'+x][id]+phot['aper_magerr_3p1_'+x][id])-8.9)/(-2.5)))
                maggies.append(m/3631)
                maggies_unc.append(m_err/3631)
            else:
                maggies.append(None)
                maggies_unc.append(None)
        
    maggies = np.array(maggies)
    maggies_unc = np.array(maggies_unc)


    #PHOT_MASK:

    for i in range(len(maggies)):
        if maggies[i] == None:
            phot_mask.append(False)
        else:
            phot_mask.append(True)
    
    phot_mask = np.array(phot_mask)


    #FILTERS:

    filternames = np.array(filternames)

#-------------------------------EMISSION LINES-------------------------------------
    
    wavelength = []
    spectrum = []
    unc = []
    mask = []


    #SPECTRUM, UNC AND WAVELENGTH:

    #for i_col in el.columns:
    #    if ('Ae' not in i_col) & ('NAME' not in i_col) & ('id' not in i_col):
    #        if el[i_col][id] == 0:
    #            spectrum.append(None)
    #            unc.append(None)
    #            wavelength.append(None)
    #            spectrum.append(el[i_col][id]*10**(-16))
    #        else:
    #            unc.append(el[i_col+'e'][id]*10**(-16))
    #            wavelength.append(translate_el[i_col])

    for i_col in el.columns:
        if ('Ae' not in i_col) & ('NAME' not in i_col) & ('id' not in i_col):
            if el[i_col][id] == 0 or i_col=='He2r_4686A':
                continue
            else:
                spectrum.append(el[i_col][id]*10**(-16))
                unc.append(el[i_col+'e'][id]*10**(-16))
                wavelength.append(translate_el[i_col])

    spectrum = np.array(spectrum)
    unc = np.array(unc)
    wavelength = np.array(wavelength)
    #unc = [spectrum[i]/10 for i in range(len(spectrum))]

    #MASK:

    for i in range(len(wavelength)):
        if wavelength[i]==None:
            mask.append(False)
        else:
            mask.append(True)

    mask = np.array(mask)

    #put the names of the lines in a list, create list with the idices of the positions our EL have in the emission_info.dat file

    line_info = np.genfromtxt('/Users/amanda/opt/anaconda3/envs/astro/lib/python3.10/site-packages/fsps/data/emlines_info.dat', dtype=[('wave', 'f8'), ('name', '<U20')], delimiter=',')
    linelist = line_info["name"].tolist()
    line_indices = []
    for n in wavelength:
        if n==None:
            continue
        else:
            line_indices.append(linelist.index(n))


    #----------------------------------NOW CREATE OBS DICTIONARY----------------------------------#

    # set up obs dictionary

    obs = {}

    # This is a list of sedpy filter objects.
    # See the sedpy.observate.load_filters command for more details on its syntax.

    obs['filters'] = load_filters(filternames)
    obs['wave_effective'] = [f.wave_effective for f in obs['filters']]

    filter_width_eff = [f.effective_width for f in obs['filters']]

    # This is a list of maggies, converted from mags.
    # It should have the same order as `filters` above.
    obs['maggies'] = maggies

    # You should use real flux uncertainties (incl. error floor)
    mags_err_final = np.clip(maggies_unc, np.abs(maggies) * err_floor, np.inf)
    obs['maggies_unc'] = mags_err_final

    # Here we mask out any NaNs or infs
    # mask bands below Lyman break (if redshift fixed)
    #obs['phot_mask'] = np.isfinite(np.squeeze(maggies)) & (mags_err < 1e4) & (mags != 1.0)
    obs['phot_mask'] = phot_mask

    # We do not have a spectrum, so we set some required elements of the obs dictionary to None.
    # (this would be a vector of vacuum wavelengths in angstroms)
    obs["wavelength"] = line_info["wave"][np.array(line_indices)]

  

    # (this would be the spectrum in units of maggies)
    obs["spectrum"] = spectrum
    obs["line_ind"] = line_indices

    # (spectral uncertainties are given here)
    #el_unc_final = np.clip(el_unc, np.abs(el_flux) * err_floor_el, np.inf)
    #obs['unc'] = el_unc_final
    obs['unc'] = unc
    # (again, to ignore a particular wavelength set the value of the
    #  corresponding elemnt of the mask to *False*)
    obs['mask'] = mask
    #obs['mask'] = [True for i in range(len(obs['spectrum']))]
    # Add unessential bonus info.  This will be stored in output
    obs['cat_row'] = id
    obs['id'] = idx_gal
    obs['z_spec'] = phot['z'][id]
    obs["line_names"] = wavelength
    return obs




def build_model_needs_thet(thet,objid=1, fit_el=True, el_table="lzlcs_optlines_obs.csv", phot_table='GP_Aperture_Matched_Photometry_v0.fits', sfh_template="continuity_sfh", add_IGM_model=False, add_duste=False, add_agn=False, add_neb=False,
                nbins_sfh=6, student_t_width=0.3,add_eline_scaling=False, **extras):
    """
    Construct a model.
    sfh_template : "continuity_sfh", "dirichlet_sfh", "parametric_sfh"

    """
    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins, adjust_dirichlet_agebins
    from prospect.models import priors
    from prospect.models import transforms

    # read in data table

    obs = build_obs(objid=objid, el_table=el_table, phot_table=phot_table, **extras)
    id = obs['cat_row']


    # get SFH template

    if (sfh_template == "continuity_sfh"):
        model_params = TemplateLibrary["continuity_sfh"]
    elif (sfh_template == "parametric_sfh"):
        model_params = TemplateLibrary["parametric_sfh"]


    # IMF: 0: Salpeter (1955); 1: Chabrier (2003); 2: Kroupa (2001)

    model_params['imf_type']['init'] = 1


    # fix redshift

    model_params["zred"]["init"] = obs['z_spec']
    model_params["zred"]["is_free"] = False
    
    # fix metalicity:

    model_params["logzsol"]["init"] = thet['Z[Z_{odot}]'][id]
    model_params["logzsol"]["is_free"] = False

    # fix escape fraction:

    model_params["frac_obrun"] = {"N": 1,
                                      "isfree": False,
                                      "init":thet['f_{esc}'][id], "units": "Fraction of H-ionizing photons that escapes or is dust absorbed."}

    # fix age:

    model_params['tage'] = {"N": 1, "isfree": False,
                                    "init": thet['AGE [Myr]'][id]/1000, "units": "Gyr"}
    

    #model_params["frac_obrun"]["init"] = 
    #model_params["frac_obrun"]["is_free"] = False


    # some functions 

    def zred_to_agebins(zred=model_params["zred"]["init"], agebins=None, z_limit_sfh=10.0, nbins_sfh=6, **extras):
        tuniv = cosmo.age(zred).value*1e9
        tbinmax = tuniv-cosmo.age(z_limit_sfh).value*1e9
        agelims = np.append(np.array([0.0, 6.7, 7.0]), np.linspace(7.0, np.log10(tbinmax), int(nbins_sfh-1))[1:])
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T
    
    def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=model_params["zred"]["init"], nbins_sfh=6, z_limit_sfh=None, **extras):
        agebins = zred_to_agebins(zred=zred, nbins_sfh=nbins_sfh)
        logsfr_ratios = np.clip(logsfr_ratios, -10, 10)  # numerical issues...
        nbins = agebins.shape[0]
        sratios = 10**logsfr_ratios
        dt = (10**agebins[:, 1]-10**agebins[:, 0])
        coeffs = np.array([(1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
        m1 = (10**logmass) / coeffs.sum()
        return m1 * coeffs

    # in the case of continuity, allow scaling with redshift (others fixed redshift)
    print('SFH template =', sfh_template)
    print('redshift init =', model_params["zred"]["init"])

    if (sfh_template == "continuity_sfh"):
        # adjust number of bins for SFH and prior
        model_params['mass']['N'] = nbins_sfh
        model_params['agebins']['N'] = nbins_sfh
        model_params['logsfr_ratios']['N'] = nbins_sfh-1
        model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1, 0.0)  # constant SFH
        model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1, 0.0), scale=np.full(nbins_sfh-1, student_t_width), df=np.full(nbins_sfh-1, 2))
        model_params['agebins']['depends_on'] = zred_to_agebins
        # set mass prior
        model_params["logmass"]["prior"] = priors.TopHat(mini=6, maxi=12)
        model_params['mass']['depends_on'] = logmass_to_masses

    elif (sfh_template == "parametric_sfh"):
        model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
        model_params["tage"]["prior"] = priors.TopHat(mini=1e-3, maxi=cosmo.age(model_params["zred"]["init"]).value)
        model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e12)


    #complexify the dust
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
    model_params["dust_index"] = {"N": 1,
                                  "isfree": True,
                                  "init": 0.0, "units": "power-law multiplication of Calzetti",
                                  "prior": priors.TopHat(mini=-1.0, maxi=0.4)}

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return(dust1_fraction*dust2)

    model_params['dust1'] = {"N": 1,
                             "isfree": False,
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # fit for IGM absorption
    if add_IGM_model:
        model_params.update(TemplateLibrary["igm"])
        model_params["igm_factor"]['isfree'] = True
        model_params["igm_factor"]["prior"] = priors.ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0)
    else:
        model_params.update(TemplateLibrary["igm"])
        model_params["igm_factor"]['isfree'] = False

    # Change the model parameter specifications based on some keyword arguments
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_umin']['isfree'] = True

    if add_agn:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['agn_tau']['isfree'] = True

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        _ = model_params["gas_logz"].pop("depends_on")
        #model_params["gas_logz"]["depends_on"] = transforms.stellar_logzsol
        # model_params['nebemlineinspec'] = {'N': 1,
        #                                    'isfree': False,
        #                                    'init': False}

    if add_eline_scaling:
        # Rescaling of emission lines
        model_params["linespec_scaling"] = {"N": 1,
                                            "isfree": True,
                                            "init": 1.0, "units": "multiplative rescaling factor",
                                            "prior": priors.ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0)}
    
    # Now instantiate the model using this new dictionary of parameter specifications
    if fit_el:
        print("FITTING EL MODEL")
        model = LineSpecModel(model_params)
    else:
        model = SpecModel(model_params)

    return model





def build_model_w(objid=0, non_param_sfh=True, dirichlet_sfh=False, stochastic_sfh=False, add_duste=False, add_neb=False, 
                marginalize_neb=False, add_agn=False, n_bins_sfh=10, fit_continuum=False, switch_off_phot=False, 
                switch_off_spec=False, fixed_dust=False, outlier_model=False, **extras):
    
    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins, adjust_dirichlet_agebins
    from prospect.models import priors
    from prospect.models import transforms
    
    obs = build_obs(objid=objid)

    
    if non_param_sfh and not dirichlet_sfh and not stochastic_sfh:
        model_params = TemplateLibrary["continuity_sfh"]
    elif dirichlet_sfh:
        model_params = TemplateLibrary["dirichlet_sfh"]
    elif stochastic_sfh:
        model_params = TemplateLibrary["stochastic_sfh"]
    else:
        model_params = TemplateLibrary["parametric_sfh"]
        
    model_params["zred"]['isfree'] = True
    model_params["zred"]["init"] = obs['z_spec']
    model_params["zred"]["prior"] = priors.TopHat(mini=obs['z_spec']-0.005, maxi=obs['z_spec']+0.005)
    
    if non_param_sfh:
        t_univ = cosmo.age(obs['z_spec']).value
        tbinmax = 0.95 * t_univ * 1e9
        #lim1, lim2, lim3, lim4 = 7.4772, 8.0, 8.5, 9.0
        # Early time bins: 1-5 Myr, 5-10, 10-30, 30-100, log...
        lim1, lim2, lim3, lim4 = 6.69897, 7.0, 7.47712, 8.0
        agelims = np.concatenate(([6., lim1, lim2], 
                                  np.log10(np.logspace(lim3, np.log10(tbinmax), n_bins_sfh-3)).flatten().tolist(), 
                                  [np.log10(t_univ*1e9)]))
        if dirichlet_sfh:
            model_params = adjust_dirichlet_agebins(model_params, agelims=agelims)
            model_params["total_mass"]["prior"] = priors.LogUniform(mini=3e9, maxi=1e12)
        
        elif stochastic_sfh:
            agebins = np.array([agelims[:-1], agelims[1:]])
            model_params['agebins']['init'] = agebins.T
            model_params["logmass"]["prior"] = priors.TopHat(mini=9.5, maxi=12.0)
            
            model_params['sigma_reg']['init'] = 0.4
            model_params['sigma_reg']['prior'] = priors.LogUniform(mini=0.1, maxi=5.0)

            model_params['tau_eq']['init'] = 2500/1e3 
            model_params['tau_eq']['prior'] = priors.TopHat(mini=0.01, maxi=t_univ)
            
            model_params['tau_in']['isfree'] = False
            model_params['tau_in']['init'] = t_univ

            model_params['sigma_dyn']['init'] = 0.03 
            model_params['sigma_dyn']['prior'] = priors.LogUniform(mini=0.001, maxi=0.1)
            
            model_params['tau_dyn']['init'] = 25/1e3
            model_params['tau_dyn']['prior'] = priors.ClippedNormal(mini=0.005, maxi=0.2, mean=0.01, sigma=0.02)
            
            model_params = adjust_stochastic_params(model_params, tuniv=t_univ)
            
        else:
            model_params = adjust_continuity_agebins(model_params, tuniv=t_univ, nbins=n_bins_sfh)
            agebins = np.array([agelims[:-1], agelims[1:]])
            model_params['agebins']['init'] = agebins.T
            model_params["logmass"]["prior"] = priors.TopHat(mini=9.5, maxi=12.0)
            
    else:
        model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=10)
        model_params["tage"]["prior"] = priors.TopHat(mini=1e-3, maxi=cosmo.age(obs['redshift']).value)
        model_params["mass"]["prior"] = priors.LogUniform(mini=3e9, maxi=1e12)
        model_params["mass"]["init_disp"] = 1e11
    
    if fixed_dust:
        model_params["logzsol"]["prior"] = priors.ClippedNormal(mini=-1.0, maxi=0.19, mean=0.0, sigma=0.15)
    else:
        model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.19)
        
    if fixed_dust:
        model_params['dust_type']['init'] = 2
        model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
        model_params['dust1'] = {"N": 1,
                                "isfree": False,
                                "init": 0.0, "units": "optical depth towards young stars",
                                "prior": None}
    else:
        model_params['dust_type']['init'] = 4
        model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
        model_params["dust_index"] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.0, "units": "power-law multiplication of Calzetti",
                                      "prior": priors.TopHat(mini=-1.0, maxi=1.0)}
        def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
            return(dust1_fraction*dust2)
        model_params['dust1'] = {"N": 1,
                                "isfree": False,
                                'depends_on': to_dust1,
                                "init": 0.0, "units": "optical depth towards young stars",
                                "prior": None}
        model_params['dust1_fraction'] = {'N': 1,
                                          'isfree': True,
                                          'init': 1.0,
                                          'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}
    
    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=40.0, maxi=400.0)

    
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = True
        model_params['duste_gamma']['init'] = 0.01
        model_params['duste_gamma']['prior'] = priors.LogUniform(mini=1e-4, maxi=0.1)
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_qpah']['prior'] = priors.TopHat(mini=0.5, maxi=7.0)
        model_params['duste_umin']['isfree'] = True
        model_params['duste_umin']['init'] = 1.0
        model_params['duste_umin']['prior'] = priors.ClippedNormal(mini=0.1, maxi=15.0, mean=2.0, sigma=1.0)

    if add_agn:
        # Allow for the presence of an AGN in the mid-infrared
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['isfree'] = True
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    if add_neb:
        # Add nebular emission
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logu']['init'] = -2.0
        model_params['gas_logz']['isfree'] = True
        _ = model_params["gas_logz"].pop("depends_on")
        
        # Adjust for widths of emission lines, i.e. gas velocity dispersion
        model_params["nebemlineinspec"]["init"] = False
        model_params["eline_sigma"] = {'N': 1, 
                                       'isfree': True, 
                                       'init': 100.0, 'units': 'km/s',
                                       'prior': priors.TopHat(mini=30, maxi=250)}
        if marginalize_neb:
            model_params.update(TemplateLibrary['nebular_marginalization'])
            model_params['eline_prior_width']['init'] = 1.0
            model_params['use_eline_prior']['init'] = True

    # This removes the continuum from the spectroscopy. Highly recommend using when modeling both photometry & spectroscopy
    if fit_continuum:
        # order of polynomial that's fit to spectrum
        #polyorder_estimate = int(np.clip(np.round((np.min([7500*(obs['redshift']+1), 9150.0])-np.max([3525.0*(obs['redshift']+1), 6000.0]))/(obs['redshift']+1)*100), 10, 30))
        model_params['polyorder'] = {'N': 1,
                                     'init': 10,
                                     'isfree': False}
        
    # This is a pixel outlier model. It helps to marginalize over poorly modeled noise, such as residual sky lines or
    # even missing absorption lines.
    if outlier_model:
        model_params['f_outlier_spec'] = {"N": 1,
                                          "isfree": True,
                                          "init": 0.01,
                                          "prior": priors.TopHat(mini=1e-5, maxi=0.5)}
        model_params['nsigma_outlier_spec'] = {"N": 1,
                                               "isfree": False,
                                               "init": 50.0}
        
    model = PolySpecModel(model_params)
    #model = sedmodel.SedModel(model_params)
    
    return model


def build_sps(zcontinuous=1, sfh_template="continuity_sfh", compute_vega_mags=False, **extras):
    if (sfh_template == "continuity_sfh") or (sfh_template == "dirichlet_sfh"):
        from prospect.sources import FastStepBasis
        sps = FastStepBasis(zcontinuous=zcontinuous,
                            compute_vega_mags=compute_vega_mags,
                            reserved_params=['tage', 'sigma_smooth'])
    else:
        from prospect.sources import CSPSpecBasis
        sps = CSPSpecBasis(zcontinuous=zcontinuous,
                           compute_vega_mags=compute_vega_mags,
                           reserved_params=['sigma_smooth'])
    return sps