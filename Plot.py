
import numpy as np
import Build
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
from toolbox_prospector import *
from dynesty.plotting import _quantile as weighted_quantile
from prospect.utils.plotting import get_best


def subcorner_custom(results, ranges, showpars=None, truths=None,
              start=0, thin=1, chains=slice(None),
              logify=["mass", "tau"], **kwargs):
    """Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset of
    the parameters.

    :param showpars: (optional)
        List of string names of parameters to include in the corner plot.

    :param truths: (optional)
        List of truth values for the chosen parameters.

    :param start: (optional, default: 0)
        The iteration number to start with when drawing samples to plot.

    :param thin: (optional, default: 1)
        The thinning of each chain to perform when drawing samples to plot.

    :param chains: (optional)
        If results are from an ensemble sampler, setting `chain` to an integer
        array of walker indices will cause only those walkers to be used in
        generating the plot.  Useful for emoving stuck walkers.

    :param kwargs:
        Remaining keywords are passed to the ``corner`` plotting package.

    :param logify:
        A list of parameter names to plot in `log10(parameter)` instead of
        `parameter`
    """
    try:
        import corner as triangle
    #except(ImportError):
    #    import triangle
    except:
        raise ImportError("Please install the `corner` package.")

    # pull out the parameter names and flatten the thinned chains
    # Get parameter names
    try:
        parnames = np.array(results['theta_labels'], dtype='U20')
    except(KeyError):
        parnames = np.array(results['model'].theta_labels())
    # Restrict to desired parameters
    if showpars is not None:
        ind_show = np.array([parnames.tolist().index(p) for p in showpars])
        parnames = parnames[ind_show]
    else:
        ind_show = slice(None)

    # Get the arrays we need (trace, wghts)
    trace = results['chain'][..., ind_show]
    if trace.ndim == 2:
        trace = trace[None, :]
    trace = trace[chains, start::thin, :]
    wghts = results.get('weights', None)
    if wghts is not None:
        wghts = wghts[start::thin]
    samples = trace.reshape(trace.shape[0] * trace.shape[1], trace.shape[2])

    # logify some parameters
    xx = samples.copy()
    if truths is not None:
        xx_truth = np.array(truths).copy()
    else:
        xx_truth = None
    for p in logify:
        if p in parnames:
            idx = parnames.tolist().index(p)
            xx[:, idx] = np.log10(xx[:, idx])
            parnames[idx] = "log({})".format(parnames[idx])
            if truths is not None:
                xx_truth[idx] = np.log10(xx_truth[idx])

    # mess with corner defaults
    corner_kwargs = {"plot_datapoints": False, "plot_density": False,
                     "fill_contours": True, "show_titles": True}
    corner_kwargs.update(kwargs)

    fig = triangle.corner(xx, labels=parnames, truths=xx_truth, range=ranges,
                          quantiles=[0.16, 0.5, 0.84], weights=wghts, **corner_kwargs)
    
    axes = fig.get_axes()
    for ax in axes:
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        
    return fig

def Corner_plots(eout, res, parnames, string, thetas, path):

    """The ranges 
    change in a way that they always incorporate the truth value"""

    for p in eout['thetas'].keys():
        q02, q98 = weighted_quantile(eout['thetas'][p]['chain'], np.array([0.02, 0.98]), weights=eout['weights'])
        for q, qstr in zip([q02, q98], ['q02', 'q98']):
            eout['thetas'][p][qstr] = q
    ranges=[]
    truths = thetas['thetas_'+string]
    for i,j in zip(parnames, range(len(parnames))):
        low = eout['thetas'][i]['q02']
        upp = eout['thetas'][i]['q98']
        dif = np.abs(np.abs(low) - np.abs(upp))
        if low <= truths[j] <= upp:
            lower = low
            upper = upp
            ranges.append((lower, upper))
        else:
            if truths[j] < low:
                lower = truths[j] - dif*0.1
                upper = upp
            else:
                upper = truths[j] + dif*0.1
                lower = low
            ranges.append((lower,upper))

    cfig = subcorner_custom(res, ranges=ranges, truths=thetas['thetas_'+string], truth_color='red', color='blue')
    #cfig = subcorner_custom(res, ranges=None,   truths=thetas['thetas_'+string], truth_color='red', color='blue')

    
    plt.savefig(path + string +'_Corner.pdf')
    plt.show()
    plt.clf()

def Res_plots(eout, res, string, path):
    tfig = reader.traceplot(res, color='blue', lw=0.07)
    plt.savefig(path+string+'_Res.pdf')
    plt.show()
    plt.clf()

# some conversions that I like to keep handy
def a2Hz(a): # a in Angstrom
    c = 2.99e18 # A/
    return c / a #in Hz
def fnu2flam(lam,fnu): # fnu in erg/s/cm2/Hz
    c = 2.998e18 #A/s
    flam = c* fnu / lam**2
    return flam
def flam2Jy(lam,flam): # flam in erg/s/cm2/A
    c = 2.998e18 #A/s
    fnu = lam**2 * flam / c # fnu in erg/s/cm2/Hz
    fnu_Jy = fnu / 1e-23
    return fnu_Jy
def flam2fnu(lam,flam): # flam in erg/s/cm2/A
    c = 2.998e18 #A/s
    fnu = lam**2 * flam / c # fnu in erg/s/cm2/Hz
    return fnu
def convertAB_Jy(AB):
    return 10**((8.9-AB)/2.5)

def Plot_Phot(num, esc, esc_out, path_output, path_plots):
    
    res, obs, model = reader.results_from(path_output+ num +'_Thetas.h5')
    if model == None:
        model   =    Build.build_model(objid=3)
    
    imax        =   np.argmax(res['lnprobability'])
    csz         =   res["chain"].shape
    theta_max   =   res['chain'][imax, :].copy()
    flatchain   =   res["chain"]

    from prospect.plotting.corner import quantile
    weights     =   res.get("weights", None)
    post_pcts   =   quantile(flatchain.T, q=[0.16, 0.50, 0.84], weights=weights)

    
    from prospect.utils.plotting import get_best
    parnames,theta_map  =   get_best(res)
    sps     =   reader.get_sps(res)
    wave    =   [f.wave_effective for f in res['obs']['filters']]
    wave    =   np.array(wave)

    f, (ax, chiax) = plt.subplots(2,1, gridspec_kw={'height_ratios': [7, 2]}, sharex=True, figsize=(16,8))

    ax.errorbar(wave,fnu2flam(wave,np.array(res['obs']['maggies'])*3631*1e-23),yerr=fnu2flam(wave, obs['maggies_unc']*3631*1e-23), label='Observed photometry',
            marker='o', markersize=3, alpha=1, ls='', lw=1,
            ecolor='red', markerfacecolor='None', markeredgecolor='red', capsize=7,
            markeredgewidth=1)

    map_parameter_vector    =   res["chain"][imax]
    spec, phot, frac        =   model.predict(map_parameter_vector, obs=obs, sps=sps)

    x   =   sps.wavelengths # restframe lambda
    y   =   fnu2flam(x,model._norm_spec*3631*1e-23)
    x2  =   wave
    y2  =   fnu2flam(x2, phot*3631*1e-23)
 
    ax.loglog(x,y, lw=0.8, color='orange', label = 'Model spectrum')
    ax.loglog(x2, y2, label='Model Photometry', 
                marker='s',markersize=10, alpha=1, ls='', lw=1,
                markerfacecolor='none', markeredgecolor='blue', 
                markeredgewidth=1)
    
    ymin    =  10**(-20.3)
    ymax    =  10**(-17.75)

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(10**(2.9), 10**(4.1))
    ax.set_xlabel(r'$\lambda_{rest-frame}$ [$\AA$]',size=16)
    ax.set_ylabel(r'f$_{\lambda}$ [cgs $\AA^{-1}$]',size=16)
    ax.set_title("Photometry, Escape Fraction: " + esc +", Modeled Escape Fraction: " + str(esc_out), size=18)

    chi_square  =   (obs['maggies'] - phot)**2 / obs['maggies_unc']**2
    chi         =   (obs['maggies'] - phot) / obs['maggies_unc']
    std=np.mean(abs(chi))

    #filter_colors = ['blue','green', 'orange', 'red','darkred', 'purple', 'c']
    filter_names    =   ['SDSS u','SDSS g','SDSS r','SDSS i','SDSS z','GALEX FUV','GALEX NUV']
    filter_position =   [3555.0, 4730.0, 6100.25, 7400.0, 8800.0, 1575.0, 2350.0]

    for f, n, x in zip(obs['filters'], filter_names, filter_position):
        w, t    =   f.wavelength.copy(), f.transmission.copy()
        t       =   t / t.max()
        t       =   10**(0.2*(np.log10(ymax/ymin)))*t * ymin

        ax.loglog(w, t, lw=1, color='gray', alpha=0.7)
        ax.fill_between(w, t, lw=1, color='gray', alpha=0.05)

    ax.legend(loc=4)

    chiax.axhspan(-1,1, color='orange', alpha=0.3, lw=0)
    chiax.axhspan(-std,std, color='white', alpha=0.9, lw=0)
    chiax.axhspan(-std,std, color='blue', alpha=0.1, lw=0)
    chiax.plot(x2,chi,'s', label='Model photometry',color='blue')
    chiax.axhline(y=0, color='red', linestyle='-', lw=0.5)
    chiax.set_ylabel('$\chi^2$ = '+str(np.round(np.sum(chi_square), 2)), size=16)

    plt.tight_layout()
    plt.savefig(path_plots+'photometry/' +num+'_Photometry.pdf', dpi=1200)


def Plot_Spec(num, esc, esc_out, dis_data, iot20, path_output, path_plot):
    res, obs, model = reader.results_from(path_output+ num +'_Thetas.h5')
    obs_e = iot20['eout_'+num]['obs']
    f = obs['spectrum']/obs_e['elines']['eline_lum']['q50'][:len(obs['spectrum'])]
    sig_dw = np.sqrt(f**2 * ((obs_e['elines']['eline_lum']['q84'][:len(obs['spectrum'])]-obs_e['elines']['eline_lum']['q50'][:len(obs['spectrum'])])/obs_e['elines']['eline_lum']['q50'][:len(obs['spectrum'])])**2 + (obs['unc']/obs['spectrum'])**2)
    sig_up = np.sqrt(f**2 * ((obs_e['elines']['eline_lum']['q50'][:len(obs['spectrum'])]-obs_e['elines']['eline_lum']['q16'][:len(obs['spectrum'])])/obs_e['elines']['eline_lum']['q50'][:len(obs['spectrum'])])**2 + (obs['unc']/obs['spectrum'])**2)
    chi =(((obs['spectrum']-obs_e['elines']['eline_lum']['q50'][:len(obs['spectrum'])]))/obs['unc'])
    chi_square  =   np.sum(chi**2) 
    chi_sum     =   np.sum(chi)

    line_info = np.genfromtxt(os.path.join(os.getenv("SPS_HOME"), "data/emlines_info.dat"), dtype=[('wave', 'f8'), ('name', '<U20')], delimiter=',')
    res['parnames'], res['thetamap'] = get_best(res)
    names_list = []
    for i in range(len(dis_data['mask'])):
        if dis_data['mask'][i]==True:
            names_list.append(line_info[obs['line_ind'][i]][1])

    fig, ax = plt.subplots(figsize=(9, 10))

    for i in range(len(obs['mask'])):
        err = np.array(sig_dw[i], sig_up[i])
        if obs['mask'][i]==True:
            ax.errorbar(f[i], i, fmt='o', color='black', xerr=err)
        else:
            ax.errorbar(f[i], i, fmt='o', color='gray', xerr=err)


    ax.set_ylim(-0.5, len(f)-0.5)
    ax.set_yticks(range(len(names_list)))
    ax.set_yticklabels(names_list, fontsize=12)
    for i in range(len(names_list)):
        ax.axhline(y=i, alpha=0.1, lw=1, color='grey')
    ax.set_title("Emission Lines, Escape Fraction: " + esc +", Modeled Escape Fraction: " + str(esc_out), size=14)
    ax.axvline(x=1.0, lw=2.0, color="red")

    ax.set_xlabel("Observation / Model", fontsize=12)
    ax.get_legend()
    ax.text(.70, 0.89, r'$\chi_{\rm tot}=%.1f$' % chi_sum, fontsize=16, ha='left', color='black', transform=ax.transAxes)
    ax.text(.70, 0.83, r'$\chi^2_{\rm square}=%.1f$' % chi_square, fontsize=16, ha='left', color='black', transform=ax.transAxes)

    #fscale = np.round([output_in['thetas']['linespec_scaling']['q50'], output_in['thetas']['linespec_scaling']['q50']-output_in['thetas']['linespec_scaling']['q16'], output_in['thetas']['linespec_scaling']['q84']-output_in['thetas']['linespec_scaling']['q50']], 3)
    #ax.text(.70, .80, r'$f_{\rm scale}=%.2f_{-%.2f}^{+%.2f}$' % (fscale[0], fscale[1], fscale[2]), fontsize=16, ha='left', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(path_plot+'spectroscopy/' +num+'_Spectroscopy.pdf', dpi=1200)