import numpy as np
import sys
import os
import pickle

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.models.sedmodel import SpecModel, LineSpecModel
from prospect.io import write_results as writer
from astropy.cosmology import Planck15 as cosmo


iot = '3'

path_wdir   =   "/Users/amanda/Desktop/Paper/technical/"
path_output =   os.path.join(path_wdir, 'prospector/input_output_test'+iot+'/')
path_data   =   os.path.join(path_wdir, "data/")

    

def build_obs(objid=3, theta = 0, err_floor=0.05, **kwargs):
    tn = str(theta)

    path_mock   =   os.path.join(path_data, 'mock/')

    with open(path_data + 'distorted_data'+tn+'.pickle', 'rb') as f:
        dis_data = pickle.load(f)

    with open(path_mock+'mock_thetas'+tn+'.pickle', 'rb') as f:
        thetas = pickle.load(f)

    print('we look at theta ' + tn + ', with params:' +str(thetas['thetas_'+tn]))
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
    return obs, tn

def build_model(objid=3, fit_el=True, el_table="lzlcs_optlines_obs.csv", phot_table='GP_Aperture_Matched_Photometry_v0.fits', sfh_template="continuity_sfh", add_frac_obrun=True, add_IGM_model=True, add_neb=True,
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
    parser.add_argument('--phot_table', type=str, default="GP_Aperture_Matched_Photometry_v0.fits",
                        help="Name of photometry table.")
    parser.add_argument('--fit_el', action="store_true",
                        help="If set, fit emission lines.")
    parser.add_argument('--el_table', type=str, default="lzlcs_optlines_obs.csv",
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
    parser.add_argument('--objid', type=int, default=3,
                        help="zero-index row number in the table to fit.")
    parser.add_argument('--theta', type=int, default=0,
                        help="which of the mock data.")
    parser.add_argument('--nbins_sfh', type=int, default=8,
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

    obs, tn, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    
    hfile = path_output + tn +  "{0}_{1}_mcmc.h5".format(args.outfile, args.objid)
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass

