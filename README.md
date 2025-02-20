# ALFAHOR - ALgorithm For Accurate H/R
Trace the vertical structure of protoplanetary disks by masking line emission. Based on Pinte et al. 2018.

# Install

You can install `ALFAHOR` using `pip`

<pre><code>pip install alfahor</pre></code>

To run the examples/tutorials you may need some files from the [MAPS collaboration](https://alma-maps.info/data.html). Any problems or questions can be sent to tpaneque@eso.org .

# How does it work?

`ALFAHOR` is a package that allows you to easily handle spectral data in FITS file format. It has an interactive interface to create masks and define visually the near and far sides of channel map emission in protoplanetary disks. Further details on the method and implementation can be found in [Paneque-Carreño et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv221001130P/abstract) and [Pinte et al. 2018](https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..47P/abstract).

# Citations

If you use `ALFAHOR` as part of your research, please cite [Paneque-Carreño et al. 2022](https://ui.adsabs.harvard.edu/abs/2022arXiv221001130P/abstract)

<pre><code>
@ARTICLE{2023A&A...669A.126P,
       author = {{Paneque-Carre{\~n}o}, T. and {Miotello}, A. and {van Dishoeck}, E.~F. and {Tabone}, B. and {Izquierdo}, A.~F. and {Facchini}, S.},
        title = "{Directly tracing the vertical stratification of molecules in protoplanetary disks}",
      journal = {\aap},
     keywords = {astrochemistry, protoplanetary disks, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2023,
        month = jan,
       volume = {669},
          eid = {A126},
        pages = {A126},
          doi = {10.1051/0004-6361/202244428},
archivePrefix = {arXiv},
       eprint = {2210.01130},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&A...669A.126P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
</pre></code>
