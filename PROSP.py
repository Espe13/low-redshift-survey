import pandas as pd
import numpy as np
import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sedpy.observate import load_filters
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.models.sedmodel import SpecModel, LineSpecModel
from prospect.sources.galaxy_basis import CSPSpecBasis
from prospect.io import write_results as writer
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
from prospect.utils.obsutils import fix_obs

#dicionaries to translate the name into the number

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



filternames = ['sdss_u0','sdss_g0','sdss_r0','sdss_i0','sdss_z0','galex_FUV','galex_NUV','wise_w1','wise_w2','wise_w3','wise_w4']
EL_info = pd.read_csv('/Users/amanda/opt/anaconda3/envs/astro/lib/python3.10/site-packages/fsps/data/emlines_info.dat', header=None, sep = ',')

#build the obs dictionary:


def build_obs(objid=3, EL_info=EL_info, el_table="lzlcs_optlines_obs.csv", phot_table='GP_Aperture_Matched_Photometry_v0.fits', err_floor=0.05, err_floor_el=0.05, filternames = filternames, **kwargs):
    
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

#build the model:

def build_model(objid=3, fit_el=False, el_table="ALL_Gal.fits", phot_table="Filter0.txt", sfh_template="continuity_sfh", add_frac_obrun=False, add_IGM_model=False, add_duste=False, add_agn=False, add_neb=False,
                nbins_sfh=6, student_t_width=0.3, z_limit_sfh=10.0, only_lowz=False, only_highz=False, add_eline_scaling=False, **extras):
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


#build sps:

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

#build noise:

def build_noise(**extras):
    return None, None

#-------------------------------------------

#build all

def build_all(**kwargs):
    return (build_obs(**kwargs), build_model(**kwargs), build_sps(**kwargs), build_noise(**kwargs))


#run it

if __name__ == '__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--phot_table', type=str, default="Filter0.txt",
                        help="Name of photometry table.")
    parser.add_argument('--fit_el', action="store_true",
                        help="If set, fit emission lines.")
    parser.add_argument('--el_table', type=str, default="ALL_Gal0_doublets.fits",
                        help="Name of emission line flux table.")
    parser.add_argument('--sfh_template', type=str, default="continuity_sfh",
                        help="SFH template assumed: continuity_sfh, dirichlet_sfh or s.")
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, add dust emission to the model.")
    #new thing
    parser.add_argument('--add_frac_obrun', action="store_true",
                        help="If set, add fraction of starlight that is not attenuated by the birth cloud dust component.")
    ##
    parser.add_argument('--add_agn', action="store_true",
                        help="If set, add AGN emission to the model.")
    parser.add_argument('--add_hard_prior_z', action="store_true",
                        help="If set, add tight redshift prior to the model.")
    parser.add_argument('--add_IGM_model', action="store_true",
                        help="If set, add flexibility to IGM model (scaling of Madau attenuation).")
    parser.add_argument('--add_eline_scaling', action="store_true",
                        help="If set, add flexibility to IGM model (scaling of Madau attenuation).")
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")
    parser.add_argument('--nbins_sfh', type=int, default=6,
                        help="Number of SFH bins.")
    parser.add_argument('--student_t_width', type=np.float64, default=0.3,
                        help="Width of student-t distribution.")
    parser.add_argument('--fix_redshift', action="store_true",
                        help="If set, the redshift is fixed to spectroscopic or photometric redshift.")
    parser.add_argument('--only_lowz', action="store_true",
                        help="If set, the redshift prior is z=1-7.")
    parser.add_argument('--only_highz', action="store_true",
                        help="If set, the redshift prior is z=7-13.")
    parser.add_argument('--err_floor', type=np.float64, default=0.05,
                        help="Error floor for photometry.")
    parser.add_argument('--err_floor_el', type=np.float64, default=0.05,
                        help="Error floor for EL.")

    args = parser.parse_args()
    run_params = vars(args)

    # add in dynesty settings
    run_params['dynesty'] = True
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_nlive_batch'] = 200
    run_params['nested_walks'] = 48  # sampling gets very inefficient w/ high S/N spectra
    run_params['nested_nlive_init'] = 2000
    run_params['nested_dlogz_init'] = 0.02
    run_params['nested_maxcall'] = 3000000
    run_params['nested_maxcall_init'] = 2000000
    run_params['nested_sample'] = 'rwalk'
    run_params['nested_maxbatch'] = None
    run_params['nested_save_bounds'] = False
    # run_params['nested_posterior_thresh'] = 0.1 old verison dynesty
    run_params['nested_target_n_effective'] = 10000
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}

    print(run_params)

    obs, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    
    hfile = path_output + "{0}_{1}_mcmc.h5".format(args.outfile, args.objid)
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
