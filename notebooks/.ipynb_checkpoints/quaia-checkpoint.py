import numpy as np
from Corrfunc import mocks
from Corrfunc.utils import convert_3d_counts_to_cf
import healpy as hp
from astropy.table import Table
from healpy.newvisufunc import projview
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.units as cu
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc import theory
from astropy.coordinates import SkyCoord

# global variables
NSIDE = 64
G_lo = 20.0
fac_stdev = 1.5

# read quaia data
def read(fn_gcatlo, name_catalog, G_lo, fn_sello, NSIDE = NSIDE, fac_stdev = fac_stdev, plot = True, cmap_map = 'plasma'):
    
    NPIX = hp.nside2npix(NSIDE)

    tab_gcatlo = Table.read(fn_gcatlo)
    
    N_gcatlo = len(tab_gcatlo)
    print(f"Number of data sources: {N_gcatlo}")

    print(tab_gcatlo.meta)

    print(f"Column names: {tab_gcatlo.columns}")

    # make map of quasar number counts
    pixel_indices_gcatlo = hp.ang2pix(NSIDE, tab_gcatlo['ra'], tab_gcatlo['dec'], lonlat=True)
    
    if plot == True:
        
        map_gcatlo = np.bincount(pixel_indices_gcatlo, minlength=NPIX)

        title_gcatlo = rf"{name_catalog}, $G<{G_lo}$ (N={len(tab_gcatlo):,})"
        projview(map_gcatlo, title=title_gcatlo,
                    unit=r"number density per healpixel (deg$^{-2}$)", cmap=cmap_map, coord=['C', 'G'], 
                    min=np.median(map_gcatlo)-fac_stdev*np.std(map_gcatlo), max=np.median(map_gcatlo)+fac_stdev*np.std(map_gcatlo), 
                    norm='log', graticule=True,
                    cbar_ticks=[5, 10, 20]) 
        
        # remove the selection function
        selfunc_lo = hp.fitsfunc.read_map(fn_sello)

        map_selfunc_lo = map_gcatlo/selfunc_lo

        title_gcatlo = rf"{name_catalog}, $G<{G_lo}$ (N={len(tab_gcatlo):,})"
        projview(map_selfunc_lo, title=title_gcatlo,
                    unit=r"number density per healpixel (deg$^{-2}$)", cmap=cmap_map, coord=['C', 'G'], 
                    min=np.nanmedian(map_selfunc_lo)-fac_stdev*np.nanstd(map_selfunc_lo), max=np.nanmedian(map_selfunc_lo)+fac_stdev*np.nanstd(map_selfunc_lo), 
                    norm='log', graticule=True,
                    cbar_ticks=[5, 20, 50]) 
        
    return tab_gcatlo, pixel_indices_gcatlo, N_gcatlo

# convert z to comoving distance in Mpc/h
def comoving_dist(z, h = 0.6844): # col 2 in fig 7 of https://arxiv.org/pdf/1807.06209
        
    H0 = h*100 * u.km/u.s/u.Mpc

    # obtain r: The comoving distance along the line-of-sight between two objects remains constant with time for objects in the Hubble flow.
    cosmo = FlatLambdaCDM(H0=H0, Om0=0.302)
    comoving_r = cosmo.comoving_distance(z)

    return (comoving_r*cu.littleh).to(u.Mpc, cu.with_H0(H0))/cu.littleh # equivalent to comoving_d*0.7

def recenter(bins):
    return 0.5*(bins[1:]+bins[:-1])

# 2D angular clustering w(theta)
def w_theta(tab_gcatlo, selfunc_lo, pixel_indices_gcatlo, tab_randlo, pixel_indices_randlo, N_gcatlo, N_randlo, RR_counts,
            thetabins = np.logspace(np.log10(0.1), np.log10(10.0), 15)):
    
    # comoving distance
    DD_counts, api_time = mocks.DDtheta_mocks(autocorr = 1, nthreads = 8, binfile = thetabins, RA1 = tab_gcatlo['ra'], 
                                          DEC1 = tab_gcatlo['dec'], weights1 = 1/selfunc_lo[pixel_indices_gcatlo],
                                          weight_type='pair_product', c_api_timer = True)
    
    # now measure clustering in random catalog
    DR_counts, api_time = mocks.DDtheta_mocks(autocorr = 0,nthreads = 8, binfile = thetabins, RA1 = tab_gcatlo['ra'], 
                                          DEC1 = tab_gcatlo['dec'], weights1 = 1/selfunc_lo[pixel_indices_gcatlo], 
                                          RA2 = tab_randlo['ra'], DEC2 = tab_randlo['dec'], 
                                          weights2 = 1/selfunc_lo[pixel_indices_randlo], weight_type='pair_product', 
                                          c_api_timer = True)
    
    # plote for sanity check
    
    DD_counts['thetaavg'] = np.mean([DD_counts['thetamin'], DD_counts['thetamax']], axis = 0)
    DR_counts['thetaavg'] = np.mean([DR_counts['thetamin'], DR_counts['thetamax']], axis = 0)

    DD_counts['npairs'] = DD_counts['npairs']*DD_counts['weightavg']
    DR_counts['npairs'] = DR_counts['npairs']*DR_counts['weightavg']
    
    # All the pair counts are done, get the angular correlation function
    return convert_3d_counts_to_cf(N_gcatlo, N_gcatlo, N_randlo, N_randlo, DD_counts, DR_counts, DR_counts, RR_counts)

# 3D projected clustering wp(rp)
def wp_rp(tab_gcatlo, tab_randlo, selfunc_lo, pixel_indices_gcatlo, pixel_indices_randlo, N_gcatlo, N_randlo, RR_counts, rmin = 0.5, rmax = 60.0, 
               nbins = 20, pimax = 40.0, d = 1):
    
    # create the bins array
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    
    # comoving distance
    DD_counts, api_time = mocks.DDrppi_mocks(autocorr = 1, cosmology = 2, nthreads = 8, pimax = pimax, binfile = rbins, 
                                         RA1 = tab_gcatlo['ra'], DEC1 = tab_gcatlo['dec'],  # where hubble distance = c/H0 and H0 = 100 km/s/Mpc h
                                         CZ1 = comoving_dist(tab_gcatlo['redshift_quaia']), weights1 = 1/selfunc_lo[pixel_indices_gcatlo],
                                         is_comoving_dist = True, weight_type='pair_product', output_rpavg = True, c_api_timer = True) # convert from Mpc to Mpc/h
    
    DR_counts, api_time = mocks.DDrppi_mocks(autocorr = 0, cosmology = 2, nthreads = 8, pimax = pimax, binfile = rbins, 
                                         RA1 = tab_gcatlo['ra'], DEC1 = tab_gcatlo['dec'], 
                                         CZ1 = comoving_dist(tab_gcatlo['redshift_quaia']), weights1 = 1/selfunc_lo[pixel_indices_gcatlo],
                                         RA2 = tab_randlo['ra'][::d], DEC2 = tab_randlo['dec'][::d],
                                         CZ2 = comoving_dist(tab_randlo['redshift_quaia'][::d]), 
                                         weights2 = 1/selfunc_lo[pixel_indices_randlo], weight_type='pair_product', 
                                         is_comoving_dist = True, output_rpavg = True, c_api_timer = True)
    
    DD_counts['npairs'] = DD_counts['npairs']*DD_counts['weightavg']
    DR_counts['npairs'] = DR_counts['npairs']*DR_counts['weightavg']
    
    # All the pair counts are done, get the angular correlation function
    wp = convert_rp_pi_counts_to_wp(N_gcatlo, N_gcatlo, int(N_randlo/d), int(N_randlo/d),
                                DD_counts, DR_counts, DR_counts, RR_counts, nbins, pimax)
    
    # calculate the x axis
    rpavg = [np.sum((DD_counts['rpavg']*DD_counts['npairs'])[DD_counts['rmin']==i])/np.sum(DD_counts['npairs'][DD_counts['rmin']==i]) 
         for i in rbins[:-1]]
    
    return wp, rpavg

# 3d clustering xi(r)
def xi_r(tab_gcatlo, tab_randlo, selfunc_lo, pixel_indices_gcatlo, pixel_indices_randlo, N_gcatlo, N_randlo, RR_counts, 
         rbins = np.logspace(np.log10(0.1), np.log10(20.0), 21)):
    
#     # obtain r: The comoving distance along the line-of-sight between two objects remains constant with time for objects in the Hubble flow.
#     ra = np.radians(tab_gcatlo['ra'].value)
#     dec = np.radians(tab_gcatlo['dec'].value)
#     r = comoving_dist(tab_gcatlo['redshift_quaia'])
    
#     # convert degrees to radians
#     X1, Y1, Z1 = r*np.cos(dec)*np.cos(ra), r*np.cos(dec)*np.sin(ra), r*np.sin(dec)
    
    c = SkyCoord(ra=tab_gcatlo['ra'].value*u.degree, dec=tab_gcatlo['dec'].value*u.degree, distance=comoving_dist(tab_gcatlo['redshift_quaia']))
    X1, Y1, Z1 = c.cartesian.xyz

    # refer to https://en.wikipedia.org/wiki/Hubble%27s_law#Dimensionless_Hubble_constant
    DD_counts, api_time = theory.DD(autocorr = 1, nthreads = 8, binfile = rbins, periodic = False,
                                X1 = X1, Y1 = Y1, Z1 = Z1, weights1 = 1/selfunc_lo[pixel_indices_gcatlo], weight_type='pair_product', 
                                output_ravg = True, c_api_timer = True) # cz/H0 = Mpc/h = m/s * km/1000 m / (100 km/s/Mpc h)

    # now measure clustering in random catalog
    c = SkyCoord(ra=tab_randlo['ra'], dec=tab_randlo['dec'], distance=comoving_dist(tab_randlo['redshift_quaia']))
    X2, Y2, Z2 = c.cartesian.xyz

    DR_counts, api_time = theory.DD(autocorr = 0, nthreads = 8, binfile = rbins, periodic = False,
                                X1 = X1, Y1 = Y1, Z1 = Z1, weights1 = 1/selfunc_lo[pixel_indices_gcatlo], X2 = X2, Y2 = Y2, Z2 = Z2,
                                weights2 = 1/selfunc_lo[pixel_indices_randlo], weight_type='pair_product', output_ravg = True, 
                                c_api_timer = True)
    
    # RR_counts, api_time = theory.DD(autocorr = 1, nthreads = 8, binfile = rbins, periodic = False,
    #                             X1 = X2, Y1 = Y2, Z1 = Z2, weights1 = 1/selfunc_lo[pixel_indices_randlo], weight_type='pair_product',
    #                             output_ravg = True, c_api_timer = True)
    
    # plot for sanity check
    DD_counts['npairs'] = DD_counts['npairs']*DD_counts['weightavg']
    DR_counts['npairs'] = DR_counts['npairs']*DR_counts['weightavg']
    # RR_counts['npairs'] = RR_counts['npairs']*RR_counts['weightavg']
    
    # All the pair counts are done, get the angular correlation function
    cf = convert_3d_counts_to_cf(N_gcatlo, N_gcatlo, N_randlo, N_randlo, DD_counts, DR_counts, DR_counts, RR_counts)
    
    return cf, DD_counts