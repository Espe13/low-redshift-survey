import numpy as np
import os
import fsps
import sys
import copy
from prospect.models import model_setup
from prospect.models.sedmodel import SpecModel
from scipy.interpolate import interp1d
from scipy.integrate import simps
from astropy import constants
from copy import deepcopy
from dynesty.plotting import _quantile as weighted_quantile
from prospect.models import sedmodel
from sedpy.observate import load_filters
from astropy.cosmology import Planck15 as cosmo
from prospect.sources.constants import lightspeed, ckms, jansky_cgs
from sedpy.observate import getSED



def build_output(res, mod, sps, obs, sample_idx, wave_spec=np.logspace(3.5, 5, 10000), ncalc=3000, slim_down=True, shorten_spec=True, non_param=False, component_nr=None, isolate_young_stars=False, time_bins_sfh=None, abslines=None, **kwargs):
    '''
    abslines = ['halpha_wide', 'halpha_narrow', 'hbeta', 'hdelta_wide', 'hdelta_narrow']
    '''

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
        spec, phot, sm = mod.predict(thetas, obs, sps=sps)
        eout['obs']['mags']['chain'][jj, :] = phot
        spec_l, phot_l, sm_l = full_model.predict(thetas, obs_l, sps=sps)
        eout['obs']['spec_l']['chain'][jj, :] = spec_l

        # emission lines
        eline_wave, eline_lum = sps.get_galaxy_elines()
        eout['obs']['elines']['eline_lum_sps']['chain'][jj, :] = eline_lum
        eout['obs']['elines']['eline_lum']['chain'][jj, :] = spec

        # calculate SFH-based quantities
        sfh_params = find_sfh_params(full_model, thetas, obs, sps, sm=sm)

        if non_param:
            sfh_params['sfh'] = -1  # non-parametric

        eout['extras']['stellar_mass']['chain'][jj] = sfh_params['mass']
        eout['extras']['stellar_mass_formed']['chain'][jj] = sfh_params['mformed']
        eout['sfh']['sfh']['chain'][jj, :] = return_full_sfh(eout['sfh']['t'], sfh_params)
        eout['extras']['time_50']['chain'][jj] = halfmass_assembly_time(sfh_params, frac_t=0.5)
        eout['extras']['time_5']['chain'][jj] = halfmass_assembly_time(sfh_params, frac_t=0.05)
        eout['extras']['time_10']['chain'][jj] = halfmass_assembly_time(sfh_params, frac_t=0.1)
        eout['extras']['time_20']['chain'][jj] = halfmass_assembly_time(sfh_params, frac_t=0.2)
        eout['extras']['time_80']['chain'][jj] = halfmass_assembly_time(sfh_params, frac_t=0.8)
        eout['extras']['time_90']['chain'][jj] = halfmass_assembly_time(sfh_params, frac_t=0.9)
        eout['extras']['tau_sf']['chain'][jj] = eout['extras']['time_20']['chain'][jj]-eout['extras']['time_80']['chain'][jj]
        eout['extras']['sfr_5']['chain'][jj] = calculate_sfr(sfh_params, 0.005,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['sfr_10']['chain'][jj] = calculate_sfr(sfh_params, 0.01,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['sfr_50']['chain'][jj] = calculate_sfr(sfh_params, 0.05,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['sfr_100']['chain'][jj] = calculate_sfr(sfh_params, 0.1,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['sfr_2000']['chain'][jj] = calculate_sfr(sfh_params, 2.0,  minsfr=-np.inf, maxsfr=np.inf)
        eout['extras']['ssfr_5']['chain'][jj] = eout['extras']['sfr_5']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        eout['extras']['ssfr_10']['chain'][jj] = eout['extras']['sfr_10']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        eout['extras']['ssfr_50']['chain'][jj] = eout['extras']['sfr_50']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        eout['extras']['ssfr_100']['chain'][jj] = eout['extras']['sfr_100']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()
        eout['extras']['ssfr_2000']['chain'][jj] = eout['extras']['sfr_2000']['chain'][jj].squeeze() / eout['extras']['stellar_mass']['chain'][jj].squeeze()

        # get spec without dust
        ndust_thetas = deepcopy(thetas)
        ndust_thetas[parnames.index('dust2')] = 0.0
        spec_l_wodust, _, _ = full_model.predict(ndust_thetas, obs_l, sps=sps)
        eout['obs']['spec_l_dustfree']['chain'][jj, :] = spec_l_wodust


        # # calculate AGN parameters if necessary
        # if 'fagn' in parnames:
        #     eout['extras']['l_agn']['chain'][jj] = measure_agn_luminosity(thetas[parnames.index('fagn')], sps, sfh_params['mformed'])

        # # lbol
        # eout['extras']['lbol']['chain'][jj] = measure_lbol(sps, sfh_params['mformed'])

        # # measure from rest-frame spectrum
        # props = measure_restframe_properties(sps, thetas=thetas, model=full_model, res=res)
        # eout['extras']['lir']['chain'][jj] = props['lir']
        # eout['extras']['luv']['chain'][jj] = props['luv']
        # eout['extras']['mag_1500']['chain'][jj] = props['mag_1500']
        # eout['extras']['lmir']['chain'][jj] = props['lmir']
        # # eout['obs']['dn4000']['chain'][jj] = props['dn4000']
        # eout['obs']['uvj'][jj, :] = props['uvj']

        # if abslines:
        #     for a in abslines:
        #         eout['obs']['abslines'][a]['flux']['chain'][jj] = props['abslines'][a]['flux']
        #         eout['obs']['abslines'][a]['ew']['chain'][jj] = props['abslines'][a]['eqw']

        # if ('igm_factor' in parnames):
        #     ndust_thetas[parnames.index('igm_factor')] = 0.0
        # props = measure_restframe_properties(sps, thetas=ndust_thetas, model=noneb_model, res=res)
        # eout['extras']['luv_intrinsic']['chain'][jj] = props['luv']
        # eout['extras']['nion']['chain'][jj] = props['nion']
        # eout['extras']['xion']['chain'][jj] = props['xion']
        # eout['extras']['mag_1500_intrinsic']['chain'][jj] = props['mag_1500']

        # nagn_thetas = deepcopy(thetas)
        # if 'fagn' in parnames:
        #     nagn_thetas[parnames.index('fagn')] = 0.0
        #     props = measure_restframe_properties(sps, thetas=nagn_thetas, model=full_model, res=res)
        #     eout['extras']['fmir']['chain'][jj] = (eout['extras']['lmir']['chain'][jj]-props['lmir'])/eout['extras']['lmir']['chain'][jj]
        #     eout['extras']['luv_agn']['chain'][jj] = props['luv']
        #     eout['extras']['lir_agn']['chain'][jj] = props['lir']

        # # isolate young star contribution
        # if isolate_young_stars:
        #     nodep_model.params['mass'] = np.zeros_like(mod.params['mass'])
        #     nodep_model.params['mass'][0] = mod.params['mass'][0]
        #     out = measure_restframe_properties(sps, model=nodep_model, thetas=nagn_thetas, res=res)
        #     eout['extras']['luv_young']['chain'][jj] = out['luv']
        #     eout['extras']['lir_young']['chain'][jj] = out['lir']

        # ages
        eout['extras']['avg_age']['chain'][jj] = massweighted_age(sfh_params)

    # calculate percentiles from chain
    for p in eout['thetas'].keys():
        q50, q16, q84 = weighted_quantile(eout['thetas'][p]['chain'], np.array([0.5, 0.16, 0.84]), weights=eout['weights'])
        for q, qstr in zip([q50, q16, q84], ['q50', 'q16', 'q84']):
            eout['thetas'][p][qstr] = q

    for p in eout['extras'].keys():
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
    for jj in range(len(obs['spectrum'])):
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
        if obs['wavelength'] is not None:
            del eout['obs']['spec_woEL']['chain']
            del eout['obs']['spec_wEL']['chain']
            del eout['obs']['spec_EL']['chain']
    return eout


def return_lir(lam, spec, z=None):
    '''
    returns IR luminosity (8-1000 microns) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    '''
    botlam = np.atleast_1d(8e4-1)
    toplam = np.atleast_1d(1000e4+1)
    edgetrans = np.atleast_1d(0)
    lir_filter = [[np.concatenate((botlam, np.linspace(8e4, 1000e4, num=100), toplam))],
                  [np.concatenate((edgetrans, np.ones(100), edgetrans))]]
    # calculate integral
    _, lir = integrate_mag(lam, spec, lir_filter, z=z)  # comes out in ergs/s
    return(lir)


def return_luv(lam, spec, z=None):
    '''
    returns UV luminosity (1216-3000 \AA) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    '''
    botlam = np.atleast_1d(1216-1)
    toplam = np.atleast_1d(3000+1)
    edgetrans = np.atleast_1d(0)
    luv_filter = [[np.concatenate((botlam, np.linspace(1216, 3000, num=100), toplam))],
                  [np.concatenate((edgetrans, np.ones(100), edgetrans))]]
    # calculate integral
    _, luv = integrate_mag(lam, spec, luv_filter, z=z)  # comes out in ergs/s
    return(luv)


def return_luv_1500(lam, spec, z=None):
    '''
    returns UV luminosity (1450-1550 \AA) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    '''
    botlam = np.atleast_1d(1450-1)
    toplam = np.atleast_1d(1550+1)
    edgetrans = np.atleast_1d(0)
    luv_filter = [[np.concatenate((botlam, np.linspace(1450, 1550, num=100), toplam))],
                  [np.concatenate((edgetrans, np.ones(100), edgetrans))]]
    # calculate integral
    absmag, luv = integrate_mag(lam, spec, luv_filter, z=z)  # comes out in ergs/s
    return(luv, absmag)


def return_Nion(lam, spec):
    '''
    returns ionizing luminosity (<912 \AA) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    '''
    lsun = 3.839e+33 #by searching charlie's github, this is in ergs/second
    hplank = 6.6261e-27 #by searching charlie's github
    c = 2.998e+18 #in angstroms/sec because we are converting wavelength in angstroms to these units
    nu = c/lam #frequency
    mask = (lam<911.6)
    nion = -np.trapz((spec/nu/hplank*lsun)[mask], x=nu[mask])
    l_uv = np.mean(spec[(lam>1450) & (lam<1550)])*lsun
    return nion, nion/l_uv


def return_lmir(lam, spec, z=None):
    '''
    returns MIR luminosity (4-20 microns) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    '''
    botlam = np.atleast_1d(4e4-1)
    toplam = np.atleast_1d(20e4+1)
    edgetrans = np.atleast_1d(0)
    lmir_filter = [[np.concatenate((botlam, np.linspace(4e4, 20e4, num=100), toplam))],
                  [np.concatenate((edgetrans, np.ones(100), edgetrans))]]
    # calculate integral
    _, lmir = integrate_mag(lam, spec, lmir_filter, z=z)  # comes out in ergs/s
    return (lmir)


def sfr_uvir(lir, luv):
    """inputs in Lsun. Calculates UV+IR SFR from Whitaker+14
    output is Msun/yr, in Chabrier IMF"""
    return 1.09e-10*(lir + 2.2*luv)


def smooth_spectrum(lam, spec, sigma, minlam=0.0, maxlam=1e50):

    '''
    ripped from Charlie Conroy's smoothspec.f90
    the 'fast way'
    integration is truncated at +/-4*sigma
    '''
    c_kms = 2.99e5
    int_trunc = 4
    spec_out = copy.copy(spec)

    for ii in range(len(lam)):
        if lam[ii] < minlam or lam[ii] > maxlam:
            spec_out[ii] = spec[ii]
            continue

        dellam = lam[ii]*(int_trunc*sigma/c_kms+1)-lam[ii]
        integrate_lam = (lam > lam[ii]-dellam) & (lam < lam[ii]+dellam)

        if np.sum(integrate_lam) <= 1:
            spec_out[ii] = spec[ii]
        else:
            vel = (lam[ii]/lam[integrate_lam]-1)*c_kms
            func = 1/np.sqrt(2*np.pi)/sigma * np.exp(-vel**2/2./sigma**2)
            dx = np.abs(np.diff(vel))  # we want absolute value
            func = func / np.trapz(func, dx=dx)
            spec_out[ii] = np.trapz(func*spec[integrate_lam], dx=dx)

    return spec_out


def find_sfh_params(model, theta, obs, sps, sm=None):
    # pass theta to model
    model.set_parameters(theta)
    # find all variables in `str_sfh_parms`
    str_sfh_params = ['sfh', 'mass', 'tau', 'sf_start', 'tage', 'sf_trunc', 'sf_slope', 'agebins', 'sfr_fraction', 'logsfr', 'const']
    sfh_out = []
    for string in str_sfh_params:
        if string in model.params:
            sfh_out.append(np.atleast_1d(model.params[string]))
        else:
            sfh_out.append(np.array([]))
    # turn into a dictionary
    iterable = [(str_sfh_params[ii], sfh_out[ii]) for ii in range(len(sfh_out))]
    out = {key: value for (key, value) in iterable}
    # if we pass stellar mass from a prior model call,
    # we don't have to calculate it here
    if sm is None and out['mass'].shape[0] == 1:
        _, _, sm_new = model.sed(theta, obs, sps=sps)
        try:
            sm = sps.csp.stellar_mass
        except AttributeError as e:
            sm = sm_new
    ### create mass fractions for nonparametric SFHs
    if out['mass'].shape[0] > 1:
        out['mass_fraction'] = out['mass']/out['mass'].sum()
        out['mformed'] = out['mass'].sum()
        out['mass'] = out['mass'].sum()*sm
    elif (model.params.get('mass_units', 'mformed') == 'mstar'):
        # If we're fitting in mstar units, swap definitions !
        out['mformed'] = out['mass'] / sm
    else:
        out['mformed'] = out['mass']
        out['mass'] *= sm
    return(out)


def test_likelihood(sps, model, obs, thetas, param_file):

    '''
    skeleton:
    load up some model, instantiate an sps, and load some observations
    generate spectrum, compare to observations, assess likelihood
    can be run in different environments as a test
    '''

    from prospect.likelihood import lnlike_spec, lnlike_phot

    run_params = model_setup.get_run_params(param_file=param_file)

    if sps is None:
        sps = model_setup.load_sps(**run_params)

    if model is None:
        model = model_setup.load_model(**run_params)

    if obs is None:
        obs = model_setup.load_obs(**run_params)

    if thetas is None:
        thetas = np.array(model.initial_theta)

    spec_noise, phot_noise = model_setup.load_gp(**run_params)

    # generate photometry
    mu, phot, x = model.mean_model(thetas, obs, sps=sps)

    # Noise modeling
    if spec_noise is not None:
        spec_noise.update(**model.params)
    if phot_noise is not None:
        phot_noise.update(**model.params)
    vectors = {'spec': mu, 'unc': obs['unc'],
               'sed': model._spec, 'cal': model._speccal,
               'phot': phot, 'maggies_unc': obs['maggies_unc']}

    # Calculate likelihoods
    lnp_prior = model.prior_product(thetas)
    lnp_spec = lnlike_spec(mu, obs=obs, spec_noise=spec_noise, **vectors)
    lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=phot_noise, **vectors)

    return lnp_prior + lnp_phot + lnp_spec


def exp_decl_sfh_half_time(tage, tau):
    ''' integrate SFR = Ae^(-t/tau)
    note that this returns YEARS AGO that the half mass was reached
    so a larger half-mass time is an OLDER galaxy
    '''
    return tage-tau*np.log(2./(1+np.exp(-tage/tau)))


def sfh_half_time(x, sfh_params, c):
    '''
    wrapper for use with halfmass assembly time
    '''
    # check for nonparametric
    if sfh_params['sf_start'].shape[0] == 0:
        sf_start = 0.0
    else:
        sf_start = sfh_params['sf_start']
    return integrate_sfh(sf_start, x, sfh_params)-c


def halfmass_assembly_time(sfh_params, frac_t=0.5):
    from scipy.optimize import brentq
    # calculate half-mass assembly time
    # c = 0.5 if half-mass assembly time occurs before burst
    try:
        half_time = brentq(sfh_half_time, 0, 14,
                           args=(sfh_params, frac_t),
                           rtol=1.48e-08, maxiter=1000)
    except ValueError:
        # big problem
        warnings.warn("You've passed SFH parameters that don't allow t_half to be calculated. Check for bugs.", UserWarning)
        half_time = np.nan
    # define age of galaxy
    tgal = sfh_params['tage']
    if tgal.shape[0] == 0:
        tgal = np.max(10**sfh_params['agebins']/1e9)
    return tgal-half_time


def integrate_mag(spec_lam, spectra, filter, z=None):
    '''
    borrowed from calc_ml
    given a filter name and spectrum, calculate magnitude/luminosity in filter (see alt_file for filter names)
    INPUT: 
        SPEC_LAM: must be in angstroms. this will NOT BE corrected for reddening even if redshift is specified. this
        allows you to calculate magnitudes in rest or observed frame.
        SPECTRA: must be in Lsun/Hz (FSPS standard). if redshift is specified, the normalization will be taken care of.
    OUTPUT:
        MAG: comes out as absolute magnitude
        LUMINOSITY: comes out in erg/s
            NOTE: if redshift is specified, INSTEAD RETURN apparent magnitude and flux [erg/s/cm^2]
    '''
    resp_lam = filter[0][0]
    res      = filter[1][0]
    # physical units, in CGS
    pc2cm = 3.08568E18
    lsun  = 3.839E33
    c     = 2.99E10
    # interpolate filter response onto spectral lambda array
    # when interpolating, set response outside wavelength range to be zero.
    response_interp_function = interp1d(resp_lam,res, bounds_error = False, fill_value = 0)
    resp_interp = response_interp_function(spec_lam)
    # integrate spectrum over filter response
    # first calculate luminosity: convert to flambda (factor of c/lam^2, with c converted to AA/s)
    # then integrate over flambda [Lsun/AA] to get Lsun
    spec_flam = spectra*(c*1e8/(spec_lam**2))
    luminosity = simps(spec_flam*resp_interp,spec_lam)
    # now calculate luminosity density [erg/s/Hz] in filter
    # this involves normalizing the filter response by integrating over wavelength
    norm = simps(resp_interp/spec_lam, spec_lam)
    luminosity_density = simps(spectra*(resp_interp/norm)/spec_lam, spec_lam)
    # if redshift is specified, convert to flux and apparent magnitude
    if z is not None:
        dfactor = (cosmo.luminosity_distance(z).value*1e5)**(-2)*(1+z)
        luminosity = luminosity*dfactor
        luminosity_density = luminosity_density*dfactor
    # convert luminosity density to flux density
    # the units of the spectra are Lsun/Hz; convert to
    # erg/s/cm^2/Hz, at 10pc for absolute mags
    flux_density = luminosity_density*lsun/(4.0*np.pi*(pc2cm*10)**2)
    luminosity   = luminosity*lsun
    # convert flux density to magnitudes in AB system
    mag = -2.5*np.log10(flux_density)-48.60
    #print 'maggies: {0}'.format(10**(-0.4*mag)*1e10)
    return(mag, luminosity)


def load_filter_response(filter, alt_file=None):
    '''READS FILTER RESPONSE CURVES FOR FSPS'''

    if not alt_file:
        filter_response_curve = os.getenv('SPS_HOME')+'/data/allfilters.dat'
    else:
        filter_response_curve = alt_file

    # initialize output arrays
    lam, res = (np.zeros(0) for i in range(2))

    # upper case?
    if filter.lower() == filter:
        lower_case = True
    else:
        lower_case = False

    # open file
    with open(filter_response_curve, 'r') as f:
        # Skips text until we find the correct filter
        for line in f:
            if lower_case:
                if line.lower().find(filter) != -1:
                    break
            else:
                if line.find(filter) != -1:
                    break
        # Reads text until we hit the next filter
        for line in f:  # This keeps reading the file
            if line.find('#') != -1:
                break
            # read line, extract data
            data = line.split()
            lam = np.append(lam, float(data[0]))
            res = np.append(res, float(data[1]))

    if len(lam) == 0:
        print("Couldn't find filter " + filter + ': STOPPING')
        print(1/0)

    return lam, res


def return_full_sfh(t, sfh_params, **kwargs):
    '''
    returns full SFH given a time vector [in Gyr] and a
    set of SFH parameters
    '''
    deltat = 1e-8  # Gyr
    # calculate new time vector such that
    # the spacing from tage back to zero
    # is identical for each SFH model
    try:
        tcalc = t-sfh_params['tage']
    except ValueError:
        tcalc = t-np.max(10**sfh_params['agebins'])/1e9
    tcalc = tcalc[tcalc < 0]*-1
    intsfr = np.zeros(len(t))
    for mm in range(len(tcalc)):
        intsfr[mm] = calculate_sfr(sfh_params, deltat, tcalc=tcalc[mm], **kwargs)
    return intsfr


def calculate_sfr(sfh_params, timescale, tcalc=None,
                  minsfr=None, maxsfr=None):
    '''
    standardized SFR calculator. returns SFR averaged over timescale.
    SFH_PARAMS: standard input
    TIMESCALE: timescale over which to calculate SFR. timescale must be in Gyr.
    TCALC: at what point in the SFH do we want the SFR? If not specified, TCALC is set to sfh_params['tage']
    MINSFR: minimum returned SFR. if not specified, minimum is 0.01% of average SFR over lifetime
    MAXSFR: maximum returned SFR. if not specified, maximum is infinite.
    returns in [Msun/yr]
    '''
    if sfh_params['sfh'] > 0:
        tage = sfh_params['tage']
    else:
        tage = np.max(10**sfh_params['agebins']/1e9)
    if tcalc is None:
        tcalc = tage
    sfr = integrate_sfh(tcalc-timescale, tcalc, sfh_params) * sfh_params['mformed'].sum()/(timescale*1e9)
    if minsfr is None:
        minsfr = sfh_params['mformed'].sum() / (tage*1e9*10000)
    if maxsfr is None:
        maxsfr = np.inf
    sfr = np.clip(sfr, minsfr, maxsfr)
    return sfr


def transform_zfraction_to_sfrfraction(zfraction):
    '''vectorized and without I/O keywords
    '''
    if zfraction.ndim == 1:
        zfraction = np.atleast_2d(zfraction).transpose()
    sfr_fraction = np.zeros_like(zfraction)
    sfr_fraction[:, 0] = 1-zfraction[:, 0]
    for i in range(1, sfr_fraction.shape[1]):
        sfr_fraction[:, i] = np.prod(zfraction[:, :i], axis=1)*(1-zfraction[:, i])
    #sfr_fraction[:,-1] = np.prod(zfraction,axis=1)  #### THIS IS SET IMPLICITLY
    return sfr_fraction


def integrate_exp_tau(t1, t2, sfh):

    return sfh['tau'][0]*(np.exp(-t1/sfh['tau'][0])-np.exp(-t2/sfh['tau'][0]))


def integrate_delayed_tau(t1, t2, sfh):

    return (np.exp(-t1/sfh['tau'])*(1+t1/sfh['tau']) - np.exp(-t2/sfh['tau'])*(1+t2/sfh['tau']))*sfh['tau']**2


def integrate_linramp(t1, t2, sfh):

    # integration constant: SFR(sf_trunc-sf_start)
    cs = (sfh['sf_trunc']-sfh['sf_start'])*(np.exp(-(sfh['sf_trunc']-sfh['sf_start'])/sfh['tau']))

    # enforce positive SFRs
    # by limiting integration to where SFR > 0
    t_zero_cross = -1.0/sfh['sf_slope'] + sfh['sf_trunc']
    if t_zero_cross > sfh['sf_trunc']-sfh['sf_start']:
        t1 = np.clip(t1, sfh['sf_trunc']-sfh['sf_start'], t_zero_cross)
        t2 = np.clip(t2, sfh['sf_trunc']-sfh['sf_start'], t_zero_cross)

    # initial integral: SFR = SFR[t=t_trunc] * [1 + s(t-t_trunc)]
    intsfr = cs*(t2-t1)*(1-sfh['sf_trunc']*sfh['sf_slope']) + cs*sfh['sf_slope']*0.5*((t2+sfh['sf_start'])**2-(t1+sfh['sf_start'])**2)

    return intsfr


def integrate_sfh(t1, t2, sfh_params):
    '''
    integrate an SFH from t1 to t2
    sfh = dictionary of SFH parameters
    returns FRACTION OF TOTAL MASS FORMED in given time inteval
    '''
    # copy so we don't overwrite values
    sfh = sfh_params.copy()
    # if we're using a parameterized SFH
    if sfh_params['sfh'] > 0:
        # make sure we have an sf_start
        if (sfh['sf_start'].shape[0] == 0):
            sfh['sf_start'] = np.atleast_1d(0.0)
        # here is our coordinate transformation to match fsps
        t1 = t1-sfh['sf_start'][0]
        t2 = t2-sfh['sf_start'][0]
        # match dimensions, if two-tau model
        ndim = len(np.atleast_1d(sfh['mass']))
        if len(np.atleast_1d(t2)) != ndim:
            t2 = np.zeros(ndim)+t2
        if len(np.atleast_1d(t1)) != ndim:
            t1 = np.zeros(ndim)+t1
        # redefine sf_trunc, if not being used for sfh=5 purposes
        if (sfh['sf_trunc'] == 0.0) or sfh['sf_trunc'].shape[0] == 0:
            sfh['sf_trunc'] = sfh['tage']
        # if we're outside of the time boundaries, clip to boundary values
        # this only affects integrals which would have been questionable in the first place
        t1 = np.clip(t1, 0, float(sfh['tage']-sfh['sf_start']))
        t2 = np.clip(t2, 0, float(sfh['tage']-sfh['sf_start']))
        # if we're using normal tau
        if (sfh['sfh'] == 1):
            # add tau model
            intsfr = integrate_exp_tau(t1, t2, sfh)
            norm = sfh['tau'][0]*(1-np.exp(-(sfh['sf_trunc'][0]-sfh['sf_start'][0])/sfh['tau'][0]))
            intsfr = intsfr/norm
        # if we're using delayed tau
        elif (sfh['sfh'] == 4):
            # add tau model
            intsfr = integrate_delayed_tau(t1, t2, sfh)
            norm = 1.0-np.exp(-(sfh['sf_trunc'][0]-sfh['sf_start'][0])/sfh['tau'][0])*(1+(sfh['sf_trunc'][0]-sfh['sf_start'][0])/sfh['tau'][0])
            intsfr = intsfr/(norm*sfh['tau'][0]**2)
        # else, add lin-ramp
        elif (sfh['sfh'] == 5):
            # by-hand calculation
            norm1 = integrate_delayed_tau(0, sfh['sf_trunc']-sfh['sf_start'], sfh)
            norm2 = integrate_linramp(sfh['sf_trunc']-sfh['sf_start'], sfh['tage']-sfh['sf_start'], sfh)
            if (t1 < sfh['sf_trunc']-sfh['sf_start']) and \
               (t2 < sfh['sf_trunc']-sfh['sf_start']):
                intsfr = integrate_delayed_tau(t1, t2, ssfh) / (norm1+norm2)
            elif (t1 > sfh['sf_trunc']-sfh['sf_start']) and \
                 (t2 > sfh['sf_trunc']-sfh['sf_start']):
                intsfr = integrate_linramp(t1, t2, sfh) / (norm1+norm2)
            else:
                intsfr = (integrate_delayed_tau(t1,sfh['sf_trunc']-sfh['sf_start'], sfh) + \
                          integrate_linramp(sfh['sf_trunc']-sfh['sf_start'], t2, sfh)) / \
                          (norm1+norm2)
        else:
            sys.exit('no such SFH implemented')
        # return sum of SFR components
        tot_mformed = np.sum(intsfr*sfh['mass'])/np.sum(sfh['mass'])
    #### nonparametric SFH
    else:
        ### make sure we got what we need
        if ('agebins' not in sfh_params):
            sys.exit('missing parameters!')
        ### put bins in proper units
        to_linear_bins = 10**sfh_params['agebins']/1e9
        time_per_bin = to_linear_bins[:, 1] - to_linear_bins[:, 0]
        time_bins = np.max(to_linear_bins) - to_linear_bins
        ### if it's outside the SFH bins, clip it
        t1 = np.clip(t1, np.min(time_bins), np.max(time_bins))
        t2 = np.clip(t2, np.min(time_bins), np.max(time_bins))
        # annoying edge cases
        if t1 == t2:
            return 0.0
        if (t2 > time_bins[0, 0]) & (t1 > time_bins[0, 0]):
            sys.exit('SFR is undefined in this time region, outside youngest bin!')
        ### which bins to integrate?
        in_range = (time_bins >= t1) & (time_bins <= t2)
        bin_ids = in_range.sum(axis=1)
        ### this doesn't work if we're fully inside a single bin...
        if in_range.sum() == 0:
            bin_ids = (time_bins[:, 1] <= t1) & (time_bins[:, 0] >= t2)
        ### weights
        weights = np.zeros(sfh_params['mass_fraction'].shape)
        # if we're all in one bin
        if np.sum(bin_ids) == 1:
            weights[bin_ids == 1] = t2-t1
        # else do the whole thing
        else:
            for i in range(bin_ids.shape[0]):
                if bin_ids[i] == 2:  # bins that are entirely in t1,t2.
                    weights[i] = time_per_bin[i]
                if bin_ids[i] == 1:  # edge cases
                    if t2 < time_bins[i, 0]:  # this is the most recent edge
                        weights[i] = t2 - time_bins[i, 1]
                    else:  # this is the oldest edge
                        weights[i] = time_bins[i, 0]-t1
                if bin_ids[i] == 0:  # no contribution
                    continue
        ### bug catch
        try:
            np.testing.assert_approx_equal(np.sum(weights), t2-t1, significant=5)
        except AssertionError:
            sys.exit('weights do not sum to 1')
        tot_mformed = np.sum((weights/time_per_bin)*sfh_params['mass_fraction'])
    return tot_mformed


def set_sfh_time_vector(res, ncalc, component_nr=None):
    """if parameterized, calculate linearly in 100 steps from t=0 to t=tage
    if nonparameterized, calculate at bin edges.
    """
    if component_nr:
        tage_label = 'tage_' + str(component_nr)
        agebins_label = 'agebins_' + str(component_nr)
    else:
        tage_label = 'tage'
        agebins_label = 'agebins'
    if tage_label in res['theta_labels']:
        nt = 100
        idx = np.array(res['theta_labels']) == tage_label
        maxtime = np.max(res['chain'][:ncalc, idx])
        t = np.linspace(0, maxtime, num=nt)
    elif agebins_label in res['model'].params:
        in_years = 10**res['model'].params[agebins_label]/1e9
        t = np.concatenate((np.ravel(in_years)*0.9999, np.ravel(in_years)*1.001))
        t.sort()
        t = t[1:-1]  # remove older than oldest bin, younger than youngest bin
        t = np.clip(t, 1e-3, np.inf)  # nothing younger than 1 Myr!
        t = np.unique(t)
    else:
        sys.exit('ERROR: not sure how to set up the time array here!')
    return t


def split_sfh_param_dict(sfh_params):
    num_comp = len(sfh_params['tage'])
    sfh_params_array = np.array([sfh_params])
    for ii_comp in range(num_comp):
        a = {}
        for ii_key in sfh_params.keys():
            if isinstance(sfh_params[ii_key], float):
                a[ii_key] = sfh_params[ii_key]
            elif (len(sfh_params[ii_key]) == 0.0):
                a[ii_key] = np.array([])
            elif (len(sfh_params[ii_key]) == 1.0):
                a[ii_key] = np.array([sfh_params[ii_key]])
            elif (len(sfh_params[ii_key]) > 1.0):
                a[ii_key] = np.array([sfh_params[ii_key][ii_comp]])
        sfh_params_array = np.append(sfh_params_array, a)
    return(sfh_params_array)


def all_ages(theta, mod, sps):
    """
    calculates light-weighted (L_bol and r-band) ages and mass-weighted ages
    all in Gyr
    """
    # zred=0
    zsave = mod.params['zred']
    mod.params['zred'] = 0.0

    # no dust
    ndust_thetas = copy.copy(theta)
    for par in ['dust1', 'dust1_fraction', 'dust2']:
        if par in mod.theta_labels():
            ndust_thetas[mod.theta_index[par]] = 0.0

    # fake obs
    fake_obs = {'maggies': None, 'phot_mask': None, 'wavelength': None, 'spectrum': None, 'filters': []}

    # Lbol light-weighted and mass-weighted ages
    sps.ssp.params['compute_light_ages'] = True
    spec, mags, sm = mod.mean_model(ndust_thetas, fake_obs, sps=sps)
    lwa_lbol = sps.ssp.log_lbol
    mwa = sps.ssp.stellar_mass

    # and this gets us the r-band light weighted age
    # this must come after the model call above, which sets tabular SFH etc
    if hasattr(sps, 'convert_sfh'):
        _, _, tmax = sps.convert_sfh(sps.params['agebins'], sps.params['mass'])
    else:
        tmax = sps.ssp.params['tage']
    lwa_rband = sps.ssp.get_mags(bands=['SDSS_r'], tage=tmax)[0]

    # restore defaults and return
    mod.params['zred'] = zsave
    sps.ssp.params['compute_light_ages'] = False

    return mwa, lwa_lbol, lwa_rband


def massweighted_age(sfh_params):
    """calculate mass-weighted age.
    currently only works for nonparameteric SFH
    """
    avg_age_per_bin = (10**sfh_params['agebins']).sum(axis=1)/2.
    mwa = (sfh_params['mass_fraction'] * avg_age_per_bin).sum()/1e9
    return mwa


def measure_lbol(sps, mass):

    '''
    requires mformed
    return in Lsun
    '''

    ## get SPS lbol, weighted by SSP weights
    # two options due to very different meanings of ssp.log_lbol when using
    # tabular or "regular" SSPs
    # THIRD OPTION: access csp
    try:
        if np.isscalar(sps.ssp.log_lbol):
            weighted_lbol = 10**sps.ssp.log_lbol
        else:
            ssp_lbol = np.insert(10**sps.ssp.log_lbol, 0, 10**sps.ssp.log_lbol[0])
            weights = sps.all_ssp_weights
            weighted_lbol = (ssp_lbol * weights).sum() / weights.sum() * mass
    except AttributeError:
        weighted_lbol = 10**sps.csp.log_lbol
    return weighted_lbol


def measure_agn_luminosity(fagn, sps, mass):

    '''
    requires mformed
    calculate L_AGN for a given F_AGN, SPS
    return in erg / s
    '''

    ## get SPS lbol, weighted by SSP weights
    # two options due to very different meanings of ssp.log_lbol when using
    # tabular or "regular" SSPs
    if np.isscalar(sps.ssp.log_lbol):
        weighted_lbol = 10**sps.ssp.log_lbol
        lagn = weighted_lbol*float(fagn)*constants.L_sun.cgs.value
    else:
        ssp_lbol = np.insert(10**sps.ssp.log_lbol, 0, 10**sps.ssp.log_lbol[0])
        weights = sps.all_ssp_weights
        weighted_lbol = (ssp_lbol * weights).sum() / weights.sum()

        ## calculate L_AGN
        lagn_sps = weighted_lbol*fagn
        lagn = lagn_sps * mass * constants.L_sun.cgs.value

    return lagn


def get_luminosity(wrest, fnu, lam_low, lam_high):
    """
    Input:
        wrest             : rest-frame wavelength in Angstrom
        fnu               : spectrum in erg / s / Hz
        lam_low, lam_high : in Angstrom
    Returns luminosity in erg / s
    """
    idx_w = (wrest > lam_low) & (wrest < lam_high)
    luminosity = -np.trapz(fnu[idx_w], lightspeed / wrest[idx_w])
    return(luminosity)


def measure_restframe_properties(sps, model=None, res=None, thetas=None):
    """measures a variety of rest-frame properties
    """
    # setup empty dict
    out = {}
    # spec is in observed frame maggies
    obs = {'filters': [], 'wavelength': None, 'spectrum': None}
    spec, phot, x = model.predict(thetas, obs=obs, sps=sps)
    # use this wavelength array assuming obs["wavelength"] = None
    wrest = sps.wavelengths
    # --- convert spectrum ---
    z = model.params['zred']
    ld = cosmo.luminosity_distance(z).to("pc").value
    # convert to maggies if the source was at 10 parsec, accounting for the (1+z) applied during predict()
    Fmaggies = spec  / (1 + z) * (ld/10)**2
    # convert to erg/s/cm^2/AA for sedpy and get absolute magnitudes
    Flambda = Fmaggies * lightspeed / wrest**2 * (3631*jansky_cgs)
    filters = ['UV1500', 'bessell_U', 'bessell_V', 'twomass_J', 'sdss_g0', 'sdss_r0', 'sdss_i0']
    obs['filters'] = load_filters(filters)
    absolute_rest_mags = getSED(wrest, Flambda, obs['filters'])
    out['mag_1500'] = absolute_rest_mags[0]
    out['uvj'] = absolute_rest_mags[1:4]
    out['sdss'] = absolute_rest_mags[4:7]
    # convert to erg/s/Hz
    pc = 3.085677581467192e18  # in cm
    dfactor_10pc = 4*np.pi*(10*pc)**2
    Fnu = Fmaggies * (3631*jansky_cgs) * dfactor_10pc
    # compute key luminosities
    out['lir'] = get_luminosity(wrest, Fnu, 8.0 * 1e4, 1000.0 * 1e4) / constants.L_sun.cgs.value
    out['lmir'] = get_luminosity(wrest, Fnu, 4.0 * 1e4, 20.0 * 1e4) / constants.L_sun.cgs.value
    out['luv'] = get_luminosity(wrest, Fnu, 1216, 3000) / constants.L_sun.cgs.value
    # compute Nion and Xion
    out['nion'], out['xion'] = return_Nion(wrest, Fnu / constants.L_sun.cgs.value)
    return(out)


def measure_Dn4000(lam, flux):
    '''
    defined as average flux ratio between
    [4050,4250] and [3750,3950] (Bruzual 1983; Hamilton 1985)
    blue: 3850-3950 . . . 4000-4100 (Balogh 1999)
    '''
    blue = (lam > 3850) & (lam < 3950)
    red = (lam > 4000) & (lam < 4100)
    dn4000 = np.mean(flux[red])/np.mean(flux[blue])
    return dn4000


def measure_emlines(smooth_spec, sps, enames=None):
    """ emission line fluxes are part of SPS output now. this is
    largely present to measure the continuum for EQW calculations
    """

    ### load fsps emission line list
    loc = os.getenv('SPS_HOME')+'/data/emlines_info.dat'
    dat = np.loadtxt(loc, delimiter=', ',
                     dtype={'names': ('lam', 'name'), 'formats': ('f16', 'S40')})

    ### define emission lines
    # legacy code compatible
    if type(enames) == bool:
        lines = np.array(['Hdelta', 'Hbeta', '[OIII]1', '[OIII]2', 'Halpha', '[NII]'])
        fsps_name = np.array(['H delta 4102', 'H beta 4861', '[OIII]4960', '[OIII]5007', 'H alpha 6563', '[NII]6585'])
    else:
        lines = enames
        fsps_name = enames

    ##### measure emission line flux + EQW
    out = {}
    for jj in range(len(lines)):

        # if we don't do nebular emission, zero this out
        if not hasattr(sps, 'get_nebline_luminosity'):
            out[lines[jj]] = {'flux': 0.0, 'eqw': 0.0}
            continue

        ### calculate luminosity (in Lsun)
        idx = (fsps_name[jj] == dat['name'])
        eflux = float(sps.get_nebline_luminosity[idx]*sps.params['mass'].sum())
        elam = float(sps.emline_wavelengths[idx])

        # simple continuum estimation
        tidx = np.abs(sps.wavelengths-elam) < 100
        eqw = eflux/np.median(smooth_spec[tidx])

        out[lines[jj]] = {'flux': eflux, 'eqw': eqw}

    return out


def measure_abslines(lam, flux):

    '''
    Nelan et al. (2005)
    Halpha wide: 6515-6540, 6554-6575, 6575-6585
    Halpha narrow: 6515-6540, 6554-6568, 6568-6575

    Worthey et al. 1994
    Hbeta: 4827.875-4847.875, 4847.875-4876.625, 4876.625-4891

    Worthey et al. 1997
    hdelta wide: 4041.6-4079.75, 4083.5-4122.25, 4128.5-4161.0
    hdelta narrow: 4057.25-4088.5, 4091-4112.25, 4114.75-4137.25

    WIDENED HALPHA AND HBETA BECAUSE OF SMOOTHING
    '''

    out = {}

    # define lines and indexes
    lines = np.array(['halpha_wide', 'halpha_narrow',
                      'hbeta',
                      'hdelta_wide', 'hdelta_narrow'])

    # improved hdelta narrow
    index = [(6540., 6586.), (6542., 6584.), (4842.875, 4884.625), (4081.5, 4124.25), (4095.0, 4113.75)]
    up = [(6590., 6610.), (6585., 6595.), (4894.625, 4910.000), (4124.25, 4151.00), (4113.75, 4130.25)]
    down = [(6515., 6540.), (6515., 6540.), (4817.875, 4835.875), (4041.6, 4081.5), (4072.25, 4094.50)]

    # measure the absorption flux
    for ii in range(len(lines)):
        dic = {}

        dic['flux'], dic['eqw'], dic['lam'] = measure_idx(lam, flux, index[ii], up[ii], down[ii])
        out[lines[ii]] = dic

    return out


def measure_idx(lam, flux, index, up, down):

    '''
    measures absorption depths
    '''

    ##### identify average flux, average wavelength
    low_cont = (lam > down[0]) & (lam < down[1])
    high_cont = (lam > up[0]) & (lam < up[1])
    abs_idx = (lam > index[0]) & (lam < index[1])

    low_flux = np.mean(flux[low_cont])
    high_flux = np.mean(flux[high_cont])

    low_lam = np.mean(down)
    high_lam = np.mean(up)

    ##### draw straight line between midpoints
    # y = mx + b
    # m = (y2 - y1) / (x2 - x1)
    # b = y0 - mx0
    m = (high_flux-low_flux)/(high_lam-low_lam)
    b = high_flux - m*high_lam

    ##### integrate the flux and the straight line, take the difference
    yline = m*lam[abs_idx]+b
    absflux = np.trapz(yline, lam[abs_idx]) - np.trapz(flux[abs_idx], lam[abs_idx])

    lamcont = np.mean(lam[abs_idx])
    abseqw = absflux/(m*lamcont+b)

    return absflux, abseqw, lamcont
