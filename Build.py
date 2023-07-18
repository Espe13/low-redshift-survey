
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


from copy import deepcopy

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from astropy.io import fits

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
from prospect.sources.galaxy_basis import CSPSpecBasis
from prospect.utils.obsutils import fix_obs
from prospect.utils.plotting import get_best

import prospect.io.read_results as reader
from toolbox_prospector import *
from prospect.models import sedmodel

from sedpy.observate import load_filters
from prospect.models.sedmodel import SpecModel, LineSpecModel
import toolbox_prospector as tp
from dynesty.plotting import _quantile as weighted_quantile



import Build


path_wdir   =   "/Users/amanda/Desktop/Paper/technical/"
path_data   =   os.path.join(path_wdir, "data/")
path_plots  =   os.path.join(path_wdir, "plots/")
path_output =   os.path.join(path_wdir, "prospector/")
path_flury  =   os.path.join(path_data, 'flury/')
path_mock   =   os.path.join(path_data, 'mock/')

with open(path_mock + 'distorted_data.pickle', 'rb') as f:
    dis_data = pickle.load(f)

theta_number = '0'


def build_dis_obs(objid=3, tn = theta_number, dis_data=dis_data, el_table="lzlcs_optlines_obs.csv", phot_table='GP_Aperture_Matched_Photometry_v0.fits', err_floor=0.05, err_floor_el=0.05, **kwargs):

    obs = {}
    obs['filters']          =   dis_data['filters']
    obs['wave_effective']   =   dis_data['wave_effective']
    obs['maggies']          =   dis_data['maggies_'+tn]
    mags_err_final          =   np.clip(dis_data['maggies_unc_'+tn], np.abs(dis_data['maggies_'+tn]) * err_floor, np.inf)
    obs['maggies_unc']      =   mags_err_final
    obs['phot_mask']        =   dis_data['phot_mask']

    obs["wavelength"]   =   dis_data['wavelength']
    obs["spectrum"]     =   dis_data['spec_'+tn]
    obs["line_ind"]     =   dis_data['line_ind']
    obs['unc']          =   dis_data['spec_unc_'+tn]
    obs['mask']         =   dis_data['mask']
    
    obs['cat_row']      =   dis_data['cat_row']
    obs['id']           =   dis_data['id']
    obs['z_spec']       =   dis_data['z_spec']
    obs["line_names"]   =   dis_data['line_names']
    return obs



def build_obs(objid=3, el_table="lzlcs_optlines_obs.csv", phot_table='GP_Aperture_Matched_Photometry_v0.fits', err_floor=0.05, err_floor_el=0.05, **kwargs):
    

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
    
    wavelength  =   []
    spectrum    =   []
    unc         =   []
    mask        =   []


    #SPECTRUM, UNC AND WAVELENGTH:

    for i_col in el.columns:
        if ('Ae' not in i_col) & ('NAME' not in i_col) & ('id' not in i_col):
            spectrum.append(el[i_col][id]*10**(-16))
            unc.append(el[i_col+'e'][id]*10**(-16))
            wavelength.append(translate_el[i_col])

    spectrum    =   np.array(spectrum)
    unc         =   np.array(unc)
    wavelength  =   np.array(wavelength)
    #unc = [spectrum[i]/10 for i in range(len(spectrum))]

    #MASK:

    for i in range(len(wavelength)):
        if wavelength[i]==None:
            mask.append(False)
        else:
            mask.append(True)

    mask = np.array(mask)

    #put the names of the lines in a list, create list with the idices of the positions our EL have in the emission_info.dat file

    line_info = np.genfromtxt(os.path.join(os.getenv("SPS_HOME"), "data/emlines_info.dat"), dtype=[('wave', 'f8'), ('name', '<U20')], delimiter=',')
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
    obs['unc']  =   unc
    # (again, to ignore a particular wavelength set the value of the
    #  corresponding elemnt of the mask to *False*)
    obs['mask'] =   mask
    #obs['mask'] = [True for i in range(len(obs['spectrum']))]
    # Add unessential bonus info.  This will be stored in output
    obs['cat_row']      =   id
    obs['id']           =   idx_gal
    obs['z_spec']       =   phot['z'][id]
    obs["line_names"]   =   wavelength
    #obs = fix_obs(obs)
    return obs




def build_model(objid=1, fit_el=True, el_table="lzlcs_optlines_obs.csv", phot_table='GP_Aperture_Matched_Photometry_v0.fits', sfh_template="continuity_sfh", add_frac_obrun=True, add_IGM_model=True, add_neb=True,
                nbins_sfh=8, student_t_width=0.3, z_limit_sfh=10.0, only_lowz=False, only_highz=False, add_eline_scaling=False, **extras):
    """
    Construct a model.
    sfh_template : "continuity_sfh", "dirichlet_sfh", "parametric_sfh"

    """
    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins, adjust_dirichlet_agebins
    from prospect.models import priors
    from prospect.models import transforms

    # read in data table
    obs = build_obs(objid=objid, el_table=el_table, phot_table=phot_table, **extras)
    #print('XXXXXXX', obs)
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
    

    def zred_to_agebins(zred=model_params["zred"]["init"], agebins=None, z_limit_sfh=10.0, nbins_sfh=8, **extras):
        tuniv = cosmo.age(zred).value*1e9
        tbinmax = tuniv-cosmo.age(z_limit_sfh).value*1e9
        agelims = np.append(np.array([0.0, 6.7, 7.0]), np.linspace(7.0, np.log10(tbinmax), int(nbins_sfh-1))[1:])
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T
    
    def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=model_params["zred"]["init"], nbins_sfh=8, z_limit_sfh=None, **extras):
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
        model_params['agebins']['N'] = nbins_sfh
        model_params['mass']['N'] = nbins_sfh
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

    # adjust other priors
    model_params["logzsol"]["prior"] = priors.ClippedNormal(mean=-1.0, sigma=0.3, mini=-2.0, maxi=0.19)  # priors.TopHat(mini=-2.0, maxi=0.19)

    # complexify the dust
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

    
    if add_frac_obrun:
        # absorb H-ionizing photons (i.e. no nebular emission)
        model_params["frac_obrun"] = {"N": 1,
                                      "isfree": True,
                                      "init": 0.0, "units": "Fraction of H-ionizing photons that escapes or is dust absorbed.",
                                      "prior": priors.ClippedNormal(mini=0.0, maxi=1.0, mean=0.0, sigma=0.5)}


    # Now instantiate the model using this new dictionary of parameter specifications
    if fit_el:
        print("FITTING EL MODEL")
        model = LineSpecModel(model_params)
    else:
        model = SpecModel(model_params)

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



def build_output(res, mod, sps, obs, sample_idx, wave_spec=np.logspace(3.5, 5, 10000), ncalc=3000, slim_down=True, shorten_spec=True, non_param=False, component_nr=None, isolate_young_stars=False, time_bins_sfh=None, abslines=None, **kwargs):
    '''
    abslines = ['halpha_wide', 'halpha_narrow', 'hbeta', 'hdelta_wide', 'hdelta_narrow']
    '''
    #obs['spectrum'] = np.array([0]*5994)

    #obs['spectrum'] = np.array([0]*5994)

    # fake obs
    obs_l = deepcopy(obs)
    obs_l['wavelength'] = wave_spec
    obs_l['unc'] = np.nan*np.ones(len(obs_l['wavelength']))
    obs_l['spectrum'] = np.nan*np.ones(len(obs_l['wavelength']))
    obs_l['mask'] = (np.ones(len(obs_l['wavelength'])) == 1)
    # compact creation of outputs
    eout = {'thetas': {},
            'extras': {},
            'sfh': {},
            'obs': {},
            'sample_idx': sample_idx,
            'weights': res['weights'][sample_idx]
            }
    fmt = {'chain': np.zeros(shape=ncalc), 'q50': 0.0, 'q84': 0.0, 'q16': 0.0}
    # thetas
    parnames = res['theta_labels']
    for i, p in enumerate(parnames):
        eout['thetas'][p] = deepcopy(fmt)
        eout['thetas'][p]['chain'] = res['chain'][sample_idx, i]
    # extras
    extra_parnames = ['evidence', 'avg_age', 'lwa_rband', 'lwa_lbol', 'time_50', 'time_5', 'time_10', 'time_20', 'time_80', 'time_90', 'tau_sf', 'sfr_5', 'sfr_10', 'sfr_50', 'sfr_100', 'sfr_2000', 'ssfr_5', 'ssfr_10', 'ssfr_50', 'ssfr_100', 'ssfr_2000', 'stellar_mass', 'stellar_mass_formed', 'lir', 'luv', 'mag_1500', 'luv_intrinsic', 'lmir', 'lbol', 'nion', 'xion', 'mag_1500_intrinsic']
    if 'fagn' in parnames:
        extra_parnames += ['l_agn', 'fmir', 'luv_agn', 'lir_agn']
    if isolate_young_stars:
        extra_parnames += ['luv_young', 'lir_young']
    for p in extra_parnames:
        eout['extras'][p] = deepcopy(fmt)
    # sfh
    if time_bins_sfh is None:
        eout['sfh']['t'] = set_sfh_time_vector(res, ncalc, component_nr=component_nr)
    else:
        eout['sfh']['t'] = time_bins_sfh
    eout['sfh']['sfh'] = {'chain': np.zeros(shape=(ncalc, eout['sfh']['t'].shape[0])), 'q50': np.zeros(shape=(eout['sfh']['t'].shape[0])), 'q84': np.zeros(shape=(eout['sfh']['t'].shape[0])), 'q16': np.zeros(shape=(eout['sfh']['t'].shape[0]))}
    # observables
    eout['obs']['mags'] = deepcopy(fmt)
    eout['obs']['uvj'] = np.zeros(shape=(ncalc, 3))
    eout['obs']['mags']['chain'] = np.zeros(shape=(ncalc, len(obs['filters'])))
    eout['obs']['spec_l'] = deepcopy(fmt)
    eout['obs']['spec_l']['chain'] = np.zeros(shape=(ncalc, len(wave_spec)))
    eout['obs']['spec_l_dustfree'] = deepcopy(fmt)
    eout['obs']['spec_l_dustfree']['chain'] = np.zeros(shape=(ncalc, len(wave_spec)))
    eout['obs']['lam_obs'] = obs['wavelength']
    eout['obs']['lam_obs_l'] = wave_spec
    if abslines:
        eout['obs']['abslines'] = {key: {'ew': deepcopy(fmt), 'flux': deepcopy(fmt)} for key in abslines}
    # eout['obs']['dn4000'] = deepcopy(fmt)
    # emission lines
    eline_wave, eline_lum = sps.get_galaxy_elines()
    eout['obs']['elines'] = {}
    eout['obs']['elines']['eline_wave'] = eline_wave
    eout['obs']['elines']['eline_lum_sps'] = deepcopy(fmt)
    eout['obs']['elines']['eline_lum_sps']['chain'] = np.zeros(shape=(ncalc, len(eline_wave)))
    eout['obs']['elines']['eline_lum'] = deepcopy(fmt)
    eout['obs']['elines']['eline_lum']['chain'] = np.zeros(shape=(ncalc, len(obs['spectrum'])))
    # generate model w/o dependencies for young star contribution
    model_params = deepcopy(mod.config_list)
    noEL_model = sedmodel.SedModel(model_params)
    for j in range(len(model_params)):
        if model_params[j]['name'] == 'mass':
            model_params[j].pop('depends_on', None)
    nodep_model = sedmodel.SedModel(model_params)
    # generate model w/o EL
    model_params = deepcopy(mod.config_list)
    for j in range(len(model_params)):
        if model_params[j]['name'] == 'add_neb_emission':
            model_params[j]['init'] = False
        elif model_params[j]['name'] == 'add_neb_continuum':
            model_params[j]['init'] = False
    noneb_model = sedmodel.SedModel(model_params)
    model_params = deepcopy(mod.config_list)
    full_model = SpecModel(model_params)


    # sample in the posterior
    for jj, sidx in enumerate(sample_idx):
        # get evidence
        eout['extras']['evidence']['chain'][jj] = res['logz'][sidx]
        # model call
        thetas = res['chain'][sidx, :]
        # get phot and spec
        print(mod)
        spec, phot, sm = mod.predict(thetas, obs, sps=sps)
        eout['obs']['mags']['chain'][jj, :] = phot
        spec_l, phot_l, sm_l = full_model.predict(thetas, obs_l, sps=sps)
        eout['obs']['spec_l']['chain'][jj, :] = spec_l
        # emission lines
        eline_wave, eline_lum = sps.get_galaxy_elines()
        eout['obs']['elines']['eline_lum_sps']['chain'][jj, :] = eline_lum
        ###DESPRATE CHANGES 2
        #eout['obs']['elines']['eline_lum']['chain'][jj, :len(spec)] = spec
        #eout['obs']['elines']['eline_lum']['chain'][jj, :] = spec
        #eout['obs']['elines']['eline_lum']['chain'][2,0:len(spec)] = eline_lum
        #eout['obs']['elines']['eline_lum']['chain'][2,len(spec):]== spec
        #eout['obs']['elines']['eline_lum']['chain'][2,0:len(spec)] = eline_lum
        
        #ORIGINAL
        # emission lines

        eline_wave, eline_lum = sps.get_galaxy_elines()
        eout['obs']['elines']['eline_lum_sps']['chain'][jj, :] = eline_lum
        eout['obs']['elines']['eline_lum']['chain'][jj, :] = spec

        # calculate SFH-based quantities
        sfh_params = tp.find_sfh_params(full_model, thetas, obs, sps, sm=sm)
        if non_param:
            sfh_params['sfh'] = -1  # non-parametric

        eout['extras']['stellar_mass']['chain'][jj] = sfh_params['mass']
        eout['extras']['stellar_mass_formed']['chain'][jj] = sfh_params['mformed']


        # eout['sfh']['sfh']['chain'][jj, :]      = return_full_sfh(eout['sfh']['t'], sfh_params)
        # eout['extras']['time_50']['chain'][jj]  = halfmass_assembly_time(sfh_params, frac_t=0.5)
        # eout['extras']['time_5']['chain'][jj]   = halfmass_assembly_time(sfh_params, frac_t=0.05)
        # eout['extras']['time_10']['chain'][jj]  = halfmass_assembly_time(sfh_params, frac_t=0.1)
        # eout['extras']['time_20']['chain'][jj]  = halfmass_assembly_time(sfh_params, frac_t=0.2)
        # eout['extras']['time_80']['chain'][jj]  = halfmass_assembly_time(sfh_params, frac_t=0.8)
        # eout['extras']['time_90']['chain'][jj]  = halfmass_assembly_time(sfh_params, frac_t=0.9)
        # eout['extras']['tau_sf']['chain'][jj]   = eout['extras']['time_20']['chain'][jj]-eout['extras']['time_80']['chain'][jj]
        # eout['extras']['sfr_5']['chain'][jj]    = calculate_sfr(sfh_params, 0.005,  minsfr=-np.inf, maxsfr=np.inf)
        # eout['extras']['sfr_10']['chain'][jj]   = calculate_sfr(sfh_params, 0.01,  minsfr=-np.inf, maxsfr=np.inf)
        # eout['extras']['sfr_50']['chain'][jj]   = calculate_sfr(sfh_params, 0.05,  minsfr=-np.inf, maxsfr=np.inf)
        # eout['extras']['sfr_100']['chain'][jj]  = calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)
        # eout['extras']['sfr_2000']['chain'][jj] = calculate_sfr(sfh_params, 2.0,  minsfr=-np.inf, maxsfr=np.inf)
        # eout['extras']['ssfr_5']['chain'][jj]   = eout['extras']['sfr_5']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        # eout['extras']['ssfr_10']['chain'][jj]  = eout['extras']['sfr_10']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        # eout['extras']['ssfr_50']['chain'][jj]  = eout['extras']['sfr_50']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        # eout['extras']['ssfr_100']['chain'][jj] = eout['extras']['sfr_100']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        # eout['extras']['ssfr_2000']['chain'][jj]= eout['extras']['sfr_2000']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
            
        # get spec without dust
        ndust_thetas = deepcopy(thetas)
        ndust_thetas[parnames.index('dust2')] = 0.0
        spec_l_wodust, _, _ = full_model.predict(ndust_thetas, obs_l, sps=sps)
        eout['obs']['spec_l_dustfree']['chain'][jj, :] = spec_l_wodust

        # ages
        eout['extras']['avg_age']['chain'][jj] = tp.massweighted_age(sfh_params)

    
    # calculate percentiles from chain
    for p in eout['thetas'].keys():
        q50, q16, q84 = weighted_quantile(eout['thetas'][p]['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
        for q, qstr in zip([q50, q16, q84], ['q50', 'q16', 'q84']):
            eout['thetas'][p][qstr] = q

    for p in eout['extras'].keys():
        if 'chain' not in eout['extras'][p]:
            continue
        elif len(eout['extras'][p]['chain']) != len(eout['weights']):
            continue
        else:
            q50, q16, q84 = weighted_quantile(eout['extras'][p]['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
            for q, qstr in zip([q50, q16, q84], ['q50', 'q16', 'q84']):
                eout['extras'][p][qstr] = q

    # q50, q16, q84 = weighted_quantile(eout['obs']['dn4000']['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    # for q, qstr in zip([q50, q16, q84], ['q50', 'q16', 'q84']):
    #     eout['obs']['dn4000'][qstr] = q

    if abslines:
        for key1 in eout['obs']['abslines'].keys():
            for key2 in ['ew', 'flux']:
                q50, q16, q84 = weighted_quantile(eout['obs']['abslines'][key1][key2]['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
                for q, qstr in zip([q50, q16, q84], ['q50', 'q16', 'q84']):
                    eout['obs']['abslines'][key1][key2][qstr] = q

    mag_pdf = np.zeros(shape=(eout['obs']['mags']['chain'].shape[1], 3))
    for jj in range(eout['obs']['mags']['chain'].shape[1]):
        mag_pdf[jj, :] = weighted_quantile(eout['obs']['mags']['chain'][:, jj], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    eout['obs']['mags']['q50'] = mag_pdf[:, 0]
    eout['obs']['mags']['q16'] = mag_pdf[:, 1]
    eout['obs']['mags']['q84'] = mag_pdf[:, 2]

    spec_pdf = np.zeros(shape=(len(eout['obs']['lam_obs_l']), 3))
    for jj in range(len(eout['obs']['lam_obs_l'])):
        spec_pdf[jj, :] = weighted_quantile(eout['obs']['spec_l']['chain'][:, jj], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    eout['obs']['spec_l']['q50'] = spec_pdf[:, 0]
    eout['obs']['spec_l']['q16'] = spec_pdf[:, 1]
    eout['obs']['spec_l']['q84'] = spec_pdf[:, 2]

    spec2_pdf = np.zeros(shape=(len(eout['obs']['lam_obs_l']), 3))
    for jj in range(len(eout['obs']['lam_obs_l'])):
        spec2_pdf[jj, :] = weighted_quantile(eout['obs']['spec_l_dustfree']['chain'][:, jj], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    eout['obs']['spec_l_dustfree']['q50'] = spec2_pdf[:, 0]
    eout['obs']['spec_l_dustfree']['q16'] = spec2_pdf[:, 1]
    eout['obs']['spec_l_dustfree']['q84'] = spec2_pdf[:, 2]

    sfh_pdf = np.zeros(shape=(eout['sfh']['t'].shape[0], 3))
    for jj in range(eout['sfh']['t'].shape[0]):
        sfh_pdf[jj, :] = weighted_quantile(eout['sfh']['sfh']['chain'][:, jj], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    eout['sfh']['sfh']['q50'] = sfh_pdf[:, 0]
    eout['sfh']['sfh']['q16'] = sfh_pdf[:, 1]
    eout['sfh']['sfh']['q84'] = sfh_pdf[:, 2]

    el_pdf = np.zeros(shape=(len(eout['obs']['elines']['eline_wave']), 3))
    el_pdf2 = np.zeros(shape=(len(eout['obs']['elines']['eline_wave']), 3))
    for jj in range(len(eout['obs']['elines']['eline_wave'])):
        el_pdf[jj, :] = weighted_quantile(eout['obs']['elines']['eline_lum_sps']['chain'][:, jj], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    for jj in range(len(spec)):
        el_pdf2[jj, :] = weighted_quantile(eout['obs']['elines']['eline_lum']['chain'][:, jj], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
    eout['obs']['elines']['eline_lum_sps']['q50'] = el_pdf[:, 0]
    eout['obs']['elines']['eline_lum_sps']['q16'] = el_pdf[:, 1]
    eout['obs']['elines']['eline_lum_sps']['q84'] = el_pdf[:, 2]
    eout['obs']['elines']['eline_lum']['q50'] = el_pdf2[:, 0]
    eout['obs']['elines']['eline_lum']['q16'] = el_pdf2[:, 1]
    eout['obs']['elines']['eline_lum']['q84'] = el_pdf2[:, 2]

    if slim_down:
        del eout['obs']['elines']['eline_lum_sps']['chain']
        del eout['obs']['elines']['eline_lum']['chain']
        del eout['extras']['stellar_mass_formed']['chain']
        del eout['extras']['lmir']['chain']
        del eout['extras']['luv']['chain']
        if isolate_young_stars:
            del eout['extras']['lir_young']['chain']
            del eout['extras']['luv_young']['chain']
        del eout['extras']['lir']['chain']
        del eout['extras']['lbol']['chain']
        del eout['sfh']['sfh']['chain']
        ###del eout['obs']['dn4000']['chain']
        del eout['obs']['spec_l']['chain']
        del eout['obs']['spec_l_dustfree']['chain']
        if 'fmir' in eout['extras'].keys():
            del eout['extras']['fmir']['chain']
        if 'lagn' in eout['extras'].keys():
            del eout['extras']['lagn']['chain']
        if 'lir_agn' in eout['extras'].keys():
            del eout['extras']['lir_agn']['chain']
        if 'luv_agn' in eout['extras'].keys():
            del eout['extras']['luv_agn']['chain']
        # if obs['wavelength'] is not None:
        #     del eout['obs']['spec_woEL']['chain']
        #     del eout['obs']['spec_wEL']['chain']
        #     del eout['obs']['spec_EL']['chain']


    obs["wavelength"] = np.linspace(1000, 1000000, num=int(1e5))

    #get prior model parameter:

    #get posterior model parameter:

    #if model==None:
    #    model = build_model()

    model_params = deepcopy(mod.config_list)
    plot_model = SpecModel(model_params)
    obs_plot = deepcopy(obs)
    obs_plot["wavelength"] = np.linspace(1000, 1000000, num=int(1e5))
    
    
    eout['model_params']    = model_params
    eout['plot_model']      = plot_model
    eout['obs_plot']        = obs_plot

    return eout
