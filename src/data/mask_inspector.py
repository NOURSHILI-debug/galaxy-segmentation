import glob
import numpy as np
from astropy.io import fits

for p in sorted(glob.glob("dataset/raw/APD-GalaxySegmentation/mask_*.fits"))[:3]:
    with fits.open(p) as h:
        print(p)
        h.info()
        for x in h:
            if x.data is None:
                continue
            d = np.asarray(x.data)
            u, c = np.unique(d, return_counts=True)
            print(d.shape, d.dtype, dict(zip(u[:20].tolist(), c[:20].tolist())), len(u), "unique")