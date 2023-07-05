"""
Lalonde-Matthews Sky Model Renderer (c)
by Lucas Valença, Jinsong Zhang, and Jean-François Lalonde.

This Lalonde-Matthews Sky Model Renderer is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""

import numpy as np
import torch
from torch import cos, acos, sin, exp

# Skylibs dependencies
from envmap import EnvironmentMap
from envmap.projections import latlong2world

class LMSkyModel(object):
    """Render panoramas according to the Lalonde-Matthews (LM) model.

    Given 11 LM parameters, this class renders linear HDR sun and sky
    panoramas, separately, in latitude-longitude format. The sun and
    sky results can be added directly to obtain the combined panorama.
    """

    def __init__(self, height, horizon_angle=90, device='cpu'):
        """Instantiates an LM renderer.

        Args:
            height:
                Integer representing the desired height (in pixels) for
                the panoramas to be rendered. The width is set as twice
                the height due to the latitude-longitude format.
            horizon_angle:
                Working range [1, 90]. Float representing the zenith
                angle (in degrees) for the horizon line. By default,
                set at the middle of the image.
            device:
                Target for PyTorch's computations.
        """
        self.height = height
        self.device = device
        self.eps = torch.finfo(torch.float32).eps

        # Getting 3D vectors for the sphere in latitude-longitude proj.
        env = EnvironmentMap(height, 'latlong')
        x, y, z, _ = latlong2world(*env.imageCoordinates())

        # Isolating the sky hemisphere given the horizon's zenith
        mask = np.arccos(y) < np.deg2rad(horizon_angle)
        self.mask = torch.tensor(mask).to(device)

        # Converting to spherical coordinates
        self.theta = self.__get_tensor(np.arccos(y)[mask])
        self.phi = self.__get_tensor(np.arctan2(x, -z)[mask])
        self.omega = self.__get_tensor(env.solidAngles()[mask])

        # Coefficients for Perez's approach as used by Preetham's model
        self.perez = self.__get_tensor([[0.1787, -1.4630],
                                        [-0.3554, 0.4275],
                                        [-0.0227, 5.3251],
                                        [0.1206, -2.5771],
                                        [-0.0670, 0.3703]])

    def get_sun(self, wsun, azimuth, zenith, beta, kappa):
        """Computes a panoramic render of the sun in the LM model.

        Can be used with either a list of parameters, generating many
        panoramas at once (e.g., with CUDA) or with single scalar
        parameters to make just one panorama.

        Inputs can be: floats, lists, NumPy arrays, or PyTorch tensors
        (in the shapes described below for each parameter).

        Args:
            wsun:
                Working range [0, inf] (usually [0, 1]). A 3D float
                vector for the HDR RGB color and intensity of the sun,
                represented as the sun weight. Can be either a len(3)
                list/array/tensor or an Nx3 list/array/tensor.
            azimuth:
                Working range [-inf, inf] (usually [-pi, pi], centered
                at 0). Float representing the azimuth angle (in radians)
                for the location of the sun pixel in spherical
                coordinates. Can be either a single scalar or a len(N)
                list/array/tensor.
            zenith:
                Working range [0, pi/2]. Float representing the zenith
                angle (in radians) for the location of the sun pixel in
                spherical coordinates. Can be either a single scalar or
                a len(N) list/array/tensor.
            beta:
                Working range [1, 50]. Float representing, together with
                kappa, the exponential falloff of sunlight as it
                scatters through the sky (as per the original paper).
                While kappa mostly impacts the size of the sun disk,
                beta mostly impacts the scattering spread. Can be either
                a single scalar or a len(N) list/array/tensor.
            kappa:
                Working range [0, 1]. Float representing, together with
                beta, the exponential falloff of sunlight as it
                scatters through the sky (as per the original paper).
                While kappa mostly impacts the size of the sun disk,
                beta mostly impacts the scattering spread. Can be either
                a single scalar or a len(N) list/array/tensor.

        Returns:
            A linear HDR panorama in the shape described when creating
            the renderer object. Values are expressed as floats.

            If a single panorama is being generated, the output shape is
            (H, W, 3) for height, width, and channels, respectively, if
            the input was a list or NumPy array. For PyTorch, the output
            is (3, H, W). If multiple sets of parameters were given,
            the output becomes (B, H, W, 3) and (B, 3, H, W) for NumPy
            and PyTorch respectively. List and NumPy inputs receive a
            NumPy output on the CPU. PyTorch inputs receive a PyTorch
            output on the input's device.
        """
        # Processing inputs
        inputs = self.__preprocess(wsun, azimuth, zenith, beta, kappa)
        is_torch, device, gamma, wsun, beta, kappa = inputs

        # Sun function (see the original paper)
        f = exp(-beta * exp(-kappa / (gamma + self.eps)))
        return self.__render(f, wsun, is_torch, device)

    def get_sky(self, wsky, azimuth, zenith, turbidity):
        """Computes a panoramic render of the sky in the LM model.

        Can be used with either a list of parameters, generating many
        panoramas at once (e.g., with CUDA) or with single scalar
        parameters to make just one panorama.

        Inputs can be: floats, lists, NumPy arrays, or PyTorch tensors
        (in the shapes described below for each parameter).

        Args:
            wsky:
                Working range [0, inf] (usually [0, 1]). A 3D float
                vector for the HDR RGB color and intensity of the sky,
                represented as the sky weight. Can be either a len(3)
                list/array/tensor or an Nx3 list/array/tensor.
            azimuth:
                Working range [-inf, inf] (usually [-pi, pi], centered
                at 0). Float representing the azimuth angle (in radians)
                for the location of the sun pixel in spherical
                coordinates. Can be either a single scalar or a len(N)
                list/array/tensor.
            zenith:
                Working range [0, pi/2]. Float representing the zenith
                angle (in radians) for the location of the sun pixel in
                spherical coordinates. Can be either a single scalar or
                a len(N) list/array/tensor.
            turbidity:
                Working range [2, 20]. Float representing the amount of
                aerosols in the sky, affecting sunlight scattering. Can
                be either a single scalar or a len(N) list/array/tensor.

        Returns:
            A linear HDR panorama in the shape described when creating
            the renderer object. Values are expressed as floats.

            If a single panorama is being generated, the output shape is
            (H, W, 3) for height, width, and channels, respectively, if
            the input was a list or NumPy array. For PyTorch, the output
            is (3, H, W). If multiple sets of parameters were given,
            the output becomes (B, H, W, 3) and (B, 3, H, W) for NumPy
            and PyTorch respectively. List and NumPy inputs receive a
            NumPy output on the CPU. PyTorch inputs receive a PyTorch
            output on the input's device.
        """
        # Processing input
        params = self.__preprocess(wsky, azimuth, zenith, turbidity)
        is_torch, device, gamma, wsky, tur = params
        tur = tur.repeat_interleave(5).view(-1, 5)

        # Perez coefficients and sky function (see the original paper)
        p = (tur * self.perez[:, 0] + self.perez[:, 1]).view(-1, 5, 1)
        f = (1 + p[:, 0] * exp(p[:, 1] / cos(self.theta) + self.eps)) *\
            (1 + p[:, 2] * exp(p[:, 3] * gamma)\
               + p[:, 4] * cos(gamma)**2)

        return self.__render(f, wsky, is_torch, device)

    def __get_tensor(self, x, filter=''):
        """Internal method that formats inputs for math operations.

        Args:
            x:
                Number or array to be formatted.
            filter:
                String indicating the type of formatting.
                If 'scalar', formats the N input scalars as Nx1.
                If 'vector', formats the N input vectors as Nx3.
                Otherwise, the shape is preserved.

        Returns:
            A PyTorch float tensor in the renderer's device, containing
            the values in x and shaped according to the filter argument.
        """
        x = torch.tensor(x) if type(x) != torch.Tensor else x
        x = x.float().to(self.device)
        if filter in ['scalar', 'vector']:
            if len(x.shape) < 2:
                x = x.unsqueeze(0)
            x = x.view(-1, 1) if filter == 'scalar' else x.view(-1, 3)
        return x

    def __preprocess(self, w_vector, azimuth, zenith, *others):
        """Formats inputs into tensors and computes the distance field.

        Args:
            w_vector:
                Weight input vector(s), either wsun or wsky.
            azimuth:
                Float(s) representing the sun's azimuth in radians.
            zenith:
                Float(s) representing the sun's zenith in radians.
            others:
                Float(s) representing the other parameters such as
                the sky's turbidity and the sun's kappa and beta.

        Returns:
            Input format information and 3 PyTorch float tensors:
                1) Information of whether the input is PyTorch or not
                   (and the device if it is).
                2) An angular distance field (in radians) from the sun's
                   pixel (in spherical coordinates) to every point in
                   the sky (as defined by the renderer's horizon line).
                3) Either wsun or wsky in the correct format.
                4) Remaining scalar parameters in the correct format.
        """
        # Input formatting
        scalar = [azimuth, zenith, *others]
        scalar = [self.__get_tensor(x, filter='scalar') for x in scalar]
        azimuth, zenith, others = (scalar[0], scalar[1], scalar[2:])

        # Storing properties
        is_torch = type(w_vector) == torch.Tensor
        device = w_vector.get_device() if is_torch else 'cpu'
        device = 'cpu' if device == -1 else device
        vector = self.__get_tensor(w_vector, filter='vector')

        # Angular distance field computation
        azimuth_cos = cos(azimuth - self.phi)
        zenith_cos = cos(zenith) * cos(self.theta)
        zenith_sin = sin(zenith) * sin(self.theta)
        gamma = acos(zenith_cos + zenith_sin * azimuth_cos)
        return is_torch, device, gamma, vector, *others

    def __render(self, f, w, is_torch, device):
        """Creates panoramas using an input function and a color weight.

        Args:
            f:
                PyTorch float tensor representing a field of function
                values for every pixel of the sky hemisphere, for either
                the sun or sky panoramas, as described in the equations
                of the original paper.
            w:
                PyTorch float tensor representing the color weight for
                the sun or sky (wsun or wsky, respectively).

        Returns:
            A linear HDR float panorama in latitude-longitude format.
        """
        # Normalization
        f_norm = (f * self.omega).sum(dim=1, keepdim=True)
        f = (f / f_norm) * (2 * np.pi)

        # Formatting function values and applying weights
        out_shape = (f.shape[0], self.height, 2 * self.height)
        out = torch.zeros(out_shape).float().to(self.device)
        out[:, self.mask] = f
        out = out.unsqueeze(1).repeat(1, 3, 1, 1)
        out = (w.view(-1, 3, 1, 1) * out)

        # Converting to input's format
        if is_torch:
            out = out.squeeze().to(device)
        else:
            out = out.permute((0, 2, 3, 1)).squeeze().cpu().numpy()
        return out
