import numpy as np
import matplotlib.pyplot as plt
from lm_model import LMSkyModel

# Engine instantiation
model = LMSkyModel(height=128, horizon_angle=90, device='cpu')

# Example of two parameter sets
params = {'wsun': [[0.79, 0.71, 0.61], [0.15, 0.13, 0.1]],
          'wsky': [[0.32, 0.42, 0.53], [0.34, 0.39, 0.43]],
          'azimuth': [0.5 * np.pi, 0],
          'zenith': [0.25 * np.pi, 0.35 * np.pi],
          'beta': [15, 5],
          'kappa': [0.025, 0.055],
          'turbidity': [2, 6.8]}

for i in range(2):
    # Example of rendering individual panoramas
    sun = model.get_sun(wsun=params['wsun'][i],
                        azimuth=params['azimuth'][i],
                        zenith=params['zenith'][i],
                        beta=params['beta'][i],
                        kappa=params['kappa'][i])

    sky = model.get_sky(wsky=params['wsky'][i],
                        azimuth=params['azimuth'][i],
                        zenith=params['zenith'][i],
                        turbidity=params['turbidity'][i])

    # Tonemapping and concatenating for display grid
    display = np.concatenate((sun, sky, sun + sky), axis=0)
    display = (display / np.percentile(display, 99))**(1/2.2)
    plt.imshow(np.clip(display, 0, 1))
    plt.show()

# Example of rendering multiple panoramas at once (full lists as input)
# can also be a PyTorch tensor or NumPy array (see documentation)
sun = model.get_sun(wsun=params['wsun'],
                    azimuth=params['azimuth'],
                    zenith=params['zenith'],
                    beta=params['beta'],
                    kappa=params['kappa'])

sky = model.get_sky(wsky=params['wsky'],
                    azimuth=params['azimuth'],
                    zenith=params['zenith'],
                    turbidity=params['turbidity'])

# Tonemapping and concatenating for display grid
display1 = np.concatenate((sun[0], sky[0], (sun + sky)[0]), axis=1)
display2 = np.concatenate((sun[1], sky[1], (sun + sky)[1]), axis=1)
display = np.concatenate((display1[:256], display2[:256]), axis=0)
display = (display / np.percentile(display, 99))**(1/2.2)
plt.imshow(np.clip(display, 0, 1))
plt.show()
