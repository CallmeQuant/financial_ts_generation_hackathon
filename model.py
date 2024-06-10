import numpy as np 
import ml_collections
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

PATH_TO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dict.pkl')
PATH_TO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_data.pkl')

# Defining Model 
# Spectral Filter
class SpectralFilter(nn.Module):
    """
    Spectral Filter torch module.
    """
    def __init__(self, d, k, FFT, hidden, flip=False, RNN=False):
        """
        Initialize the SpectralFilter module.

        Parameters
        ----------
        d : int
            Number of input dimensions.
        k : int
            Dimension of split in the input space.
        FFT : int
            Number of FFT components.
        hidden : int
            Number of hidden units in the spectral filter layer.
        flip : bool, optional
            Indicator on whether to flip the split dimensions (default is False).
        RNN : bool, optional
            Indicator on whether to use an RNN in spectral filtering (default is False).
        """
        super().__init__()

        self.d, self.k = d, k

        self.out_size = self.d - self.k
        self.pz_size = self.d
        self.in_size = self.k

        if flip:
            self.in_size, self.out_size = self.out_size, self.in_size

        self.sig_net = nn.Sequential(
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.out_size),
        )

        self.mu_net = nn.Sequential(
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.out_size),
        )

        base_mu, base_cov = torch.zeros(self.pz_size), torch.eye(self.pz_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x, flip=False):
        """
        Forward pass of the spectral filter.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        flip : bool, optional
            Indicator on whether to flip the split dimensions (default is False).

        Returns
        -------
        z_hat : torch.Tensor
            Transformed tensor.
        log_pz : torch.Tensor
            Log probability of the transformed tensor.
        log_jacob : torch.Tensor
            Log Jacobian determinant.
        """
        x1, x2 = x[:, :self.k], x[:, self.k:]

        if flip:
            x2, x1 = x1, x2

        sig = self.sig_net(x1).view(-1, self.out_size)
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(x1).view(-1, self.out_size)

        if flip:
            z2, z1 = z1, z2

        z_hat = torch.cat([z1, z2], dim=-1)

        log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        return z_hat, log_pz, log_jacob

    def inverse(self, Z, flip=False):
        """
        Inverse pass of the spectral filter.

        Parameters
        ----------
        Z : torch.Tensor
            Transformed tensor.
        flip : bool, optional
            Indicator on whether to flip the split dimensions (default is False).

        Returns
        -------
        x : torch.Tensor
            Reconstructed input tensor.
        """
        z1, z2 = Z[:, :self.k], Z[:, self.k:]

        if flip:
            z2, z1 = z1, z2

        x1 = z1

        sig_in = self.sig_net(z1).view(-1, self.out_size)
        x2 = (z2 - self.mu_net(z1).view(-1, self.out_size)) * torch.exp(-sig_in)

        if flip:
            x2, x1 = x1, x2

        return torch.cat([x1, x2], -1)


def flip(x, dim):
    """
    Flipping helper.

    Takes a vector as an input, then flips its elements from left to right.

    Parameters
    ----------
    x : torch.Tensor
        Input vector of size N x 1.
    dim : int
        Splitting dimension.

    Returns
    -------
    torch.Tensor
        Flipped vector.
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)

def reconstruct_DFT(x, component="real"):
    """
    Prepares input for the DFT inverse.

    Takes a cropped frequency and creates a symmetric or anti-symmetric mirror of it before applying inverse DFT.

    Parameters
    ----------
    x : torch.Tensor
        Cropped frequency tensor.
    component : str, optional
        Component type, either "real" or "imag" (default is "real").

    Returns
    -------
    torch.Tensor
        Reconstructed frequency tensor.
    """
    if component == "real":
        x_rec = torch.cat([x[0, :], flip(x[0, :], dim=0)], dim=0)
    elif component == "imag":
        x_rec = torch.cat([x[1, :], -1 * flip(x[1, :], dim=0)], dim=0)
    return x_rec

class DFT(nn.Module):
    """
    Discrete Fourier Transform (DFT) torch module.

    Attributes
    ----------
    N_fft : int
        Size of the DFT transform, conventionally set to the length of the input time-series or a fixed number of desired spectral components.
    crop_size : int
        Size of non-redundant frequency components, i.e., N_fft / 2 since we deal with real-valued inputs and the DFT is symmetric around 0.
    base_dist : torch.distributions.MultivariateNormal
        Base distribution of the flow, always defined as a multivariate normal distribution.
    """

    def __init__(self, N_fft=100):
        """
        Initialize the DFT module.

        Parameters
        ----------
        N_fft : int, optional
            Size of the DFT transform (default is 100).
        """
        super(DFT, self).__init__()

        self.N_fft = N_fft
        self.crop_size = int(np.ceil(self.N_fft / 2))
        base_mu, base_cov = torch.zeros(self.crop_size * 2), torch.eye(self.crop_size * 2)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x):
        """
        Forward pass of the DFT.

        Steps
        -----
        1. Convert the input vector to numpy format.
        2. Apply FFT in numpy with FFTshift to center the spectrum around 0.
        3. Crop the spectrum by removing half of the real and imaginary components. Note that the FFT output size
           is 2 * N_fft because the DFT output is complex-valued rather than real-valued. After cropping, the size
           remains N_fft, similar to the input time-domain signal. In this step we also normalize the spectrum by N_fft.
        4. Convert spectrum back to torch tensor format.
        5. Compute the flow likelihood and Jacobian. Because DFT is a Vandermonde linear transform, Log-Jacob-Det = 0.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x_fft : torch.Tensor
            Transformed frequency tensor.
        log_pz : torch.Tensor
            Log probability of the transformed tensor.
        log_jacob : int
            Log Jacobian determinant, which is 0.
        """
        if len(x.shape) == 1:
            x = x.reshape((1, -1))

        x_numpy = x.detach().float()
        X_fft = [np.fft.fftshift(np.fft.fft(x_numpy[k, :])) for k in range(x.shape[0])]
        X_fft_train = np.array(
            [
                np.array(
                    [
                        np.real(X_fft[k])[:self.crop_size] / self.N_fft,
                        np.imag(X_fft[k])[:self.crop_size] / self.N_fft,
                    ]
                )
                for k in range(len(X_fft))
            ]
        )
        x_fft = torch.from_numpy(X_fft_train).float()

        log_pz = self.base_dist.log_prob(
            x_fft.view(-1, x_fft.shape[1] * x_fft.shape[2])
        )
        log_jacob = 0

        return x_fft, log_pz, log_jacob

    def inverse(self, x):
        """
        Inverse pass of the DFT.

        Steps
        -----
        1. Convert the input vector to numpy format with size NUM_SAMPLES x 2 x N_fft.
           Second dimension indexes the real and imaginary components.
        2. Apply FFT in numpy with FFTshift to center the spectrum around 0.
        3. Crop the spectrum by removing half of the real and imaginary components. Note that the FFT output size
           is 2 * N_fft because the DFT output is complex-valued rather than real-valued. After cropping, the size
           remains N_fft, similar to the input time-domain signal. In this step we also normalize the spectrum by N_fft.
        4. Convert spectrum back to torch tensor format.
        5. Compute the flow likelihood and Jacobian. Because DFT is a Vandermonde linear transform, Log-Jacob-Det = 0.

        Parameters
        ----------
        x : torch.Tensor
            Transformed tensor.

        Returns
        -------
        x_ifft_out : torch.Tensor
            Reconstructed input tensor.
        """
        x_numpy = x.view((-1, 2, self.crop_size))

        x_numpy_r = [
            reconstruct_DFT(x_numpy[u, :, :], component="real").detach().numpy()
            for u in range(x_numpy.shape[0])
        ]
        x_numpy_i = [
            reconstruct_DFT(x_numpy[u, :, :], component="imag").detach().numpy()
            for u in range(x_numpy.shape[0])
        ]

        x_ifft = [
            self.N_fft
            * np.real(np.fft.ifft(np.fft.ifftshift(x_numpy_r[u] + 1j * x_numpy_i[u])))
            for u in range(x_numpy.shape[0])
        ]
        x_ifft_out = torch.from_numpy(np.array(x_ifft)).float()

        return x_ifft_out

def calculate_correlation_difference(real_data, generated_data):
    """
    Calculate the L1 norm of the difference between the correlation matrices
    of real and generated data.

    Parameters
    ----------
    real_data : torch.Tensor
        Real data tensor of shape (batch_size, seq_len, num_feats).
    generated_data : torch.Tensor
        Generated data tensor of shape (batch_size, seq_len, num_feats).

    Returns
    -------
    torch.Tensor
        L1 norm of the correlation difference.
    """
    # Merge the last two dimensions
    real_data = real_data.view(real_data.size(0), -1)
    generated_data = generated_data.view(generated_data.size(0), -1)
    
    # Compute correlation matrices
    real_corr = torch.corrcoef(real_data.T)
    generated_corr = torch.corrcoef(generated_data.T)
    
    return F.l1_loss(real_corr, generated_corr)


def calculate_acf(x, max_lag):
    """
    Calculate the auto-correlation function up to max_lag for each series in the batch.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, num_feats).
    max_lag : int
        Maximum lag to calculate the ACF.

    Returns
    -------
    acf : torch.Tensor
        ACF tensor of shape (batch_size, max_lag + 1).
    """
    # Merge the last two dimensions
    x = x.view(x.size(0), -1)

    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, unbiased=False, keepdim=True)
    x_centered = x - mean

    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            cov = torch.mean(x_centered * x_centered, dim=1)
        else:
            cov = torch.mean(x_centered[:, :-lag] * x_centered[:, lag:], dim=1)
        acf.append(cov / var.squeeze())
    return torch.stack(acf, dim=1)

def calculate_acf_difference(real_data, generated_data, max_lag):
    """
    Calculate the L1 norm of the difference between the auto-correlation functions
    of real and generated data.

    Parameters
    ----------
    real_data : torch.Tensor
        Real data tensor of shape (batch_size, seq_len, num_feats).
    generated_data : torch.Tensor
        Generated data tensor of shape (batch_size, seq_len, num_feats).
    max_lag : int
        Maximum lag to calculate the ACF.

    Returns
    -------
    torch.Tensor
        L1 norm of the ACF difference.
    """
    real_acf = calculate_acf(real_data, max_lag)
    fake_acf = calculate_acf(generated_data, max_lag)
    return F.l1_loss(fake_acf, real_acf)


class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, batch_size: int, n_lags: int, device: str):
        """ Implement here generation scheme. """
        pass

class FourierFlow(GeneratorBase):
    def __init__(self, input_dim, output_dim, hidden, n_flows, n_lags,
                 vol_activation='softplus', FFT=True, flip=True, normalize=False):
        super(FourierFlow, self).__init__(input_dim, output_dim)

        self.FFT = FFT
        self.normalize = normalize
        self.n_flows = n_flows
        self.hidden = hidden
        self.output_dim = output_dim
        self.individual_shape = (n_lags, output_dim)
        self.d = np.prod(self.individual_shape)
        self.k = int(np.ceil(self.d / 2))

        # Activation function for volatility
        if vol_activation == 'relu':
            self.activation_fn = self.relu
        elif vol_activation == 'softplus':
            self.activation_fn = self.softplus
        else:
            raise ValueError("Unsupported activation. Choose 'relu' or 'softplus'.")

        if flip:
            self.flips = [True if i % 2 else False for i in range(n_flows)]
        else:
            self.flips = [False for i in range(n_flows)]

    def forward(self, batch_size: int, n_lags: int, device: str):
        """
        Generate samples using the trained model.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        n_lags : int
            Length of the sequence to generate.
        device : str
            Device to use for computation ('cpu' or 'cuda').

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        return self.sample(batch_size).to(device)

    def forward_step(self, x):
        """
        Perform one step of the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor.
        log_pz : torch.Tensor
            Log probability of the transformed tensor.
        log_jacob : float
            Sum of log determinants of the Jacobian matrices.
        """
        if self.FFT:
            x = self.FourierTransform(x)[0]
            if self.normalize:
                x = (x - self.fft_mean) / (self.fft_std + 1e-8)
            x = x.view(-1, self.d)

        log_jacobs = []
        for bijector, f in zip(self.bijectors, self.flips):
            x, log_pz, lj = bijector(x, flip=f)
            log_jacobs.append(lj)
        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):
        """
        Perform the inverse transformation.

        Parameters
        ----------
        z : torch.Tensor
            Transformed tensor.

        Returns
        -------
        numpy.ndarray
            Reconstructed input tensor.
        """
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=f)
        if self.FFT:
            if self.normalize:
                z = z * self.fft_std.view(-1, self.d) + self.fft_mean.view(-1, self.d)
            z = self.FourierTransform.inverse(z)
        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        epochs : int, optional
            Number of training epochs (default is 500).
        batch_size : int, optional
            Batch size (default is 128).
        learning_rate : float, optional
            Learning rate (default is 1e-3).
        display_step : int, optional
            Interval to display training progress (default is 100).

        Returns
        -------
        list
            Training losses.
        """
        X_train = torch.from_numpy(np.array(X)).float()

        self.carry_flag = False
        if np.prod(X_train.shape[1:]) % 2 == 1:
            repeat_last = X_train[:, :, -1:]
            X_train = torch.cat([X_train, repeat_last], dim=2)
            self.carry_flag = True

        self.individual_shape = X_train.shape[1:]
        self.d = np.prod(self.individual_shape)
        self.k = int(np.ceil(self.d / 2))

        assert self.d % 2 == 0

        self.bijectors = nn.ModuleList(
            [
                SpectralFilter(self.d, self.k, self.FFT, hidden=self.hidden, flip=self.flips[_])
                for _ in range(self.n_flows)
            ]
        )

        self.FourierTransform = DFT(N_fft=self.d)
        X_train = X_train.reshape(-1, self.d)

        X_train_spectral = self.FourierTransform(X_train)[0]
        assert X_train_spectral.shape[-1] == self.k

        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

        losses = []

        for step in tqdm(range(epochs), desc="Training Epochs"):
            optimizer.zero_grad()

            z, log_pz, log_jacob = self.forward_step(X_train)
            base_loss = (-log_pz - log_jacob).mean()

            generated_data = self.inverse(z)
            generated_data = torch.tensor(generated_data).reshape(-1, *self.individual_shape)

            if self.carry_flag:
                generated_data = generated_data[:, :, :-1]

            real_data = X_train.reshape(-1, *self.individual_shape)

            correlation_loss = calculate_correlation_difference(real_data.view(real_data.shape[0], -1), 
                                                                generated_data.view(generated_data.shape[0], -1))
            acf_loss = calculate_acf_difference(real_data.view(real_data.shape[0], -1), 
                                                generated_data.view(generated_data.shape[0], -1), max_lag=5)

            total_loss = base_loss + correlation_loss + acf_loss
            losses.append(total_loss.detach().numpy())

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if (step % display_step == 0) or (step == epochs - 1):
                print(f"{step}/{epochs} | Base Loss: {base_loss.item():.3f} | Correlation Loss: {correlation_loss.item():.3f} \
                | ACF Loss: {acf_loss.item():.3f} \
                | Total Loss: {total_loss.item():.3f}")

        return losses

    def sample(self, n_samples, device='cpu'):
        """
        Sample new data points from the trained model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        device : str, optional
            Device to use for computation (default is 'cpu').

        Returns
        -------
        torch.Tensor
            Generated samples.
        """
        mu, cov = torch.zeros(self.d), torch.eye(self.d)
        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,)).to(device)

        X_sample = self.inverse(z)
        X_sample = torch.tensor(X_sample).reshape(-1, *self.individual_shape).to(device)

        if self.carry_flag:
            X_sample = X_sample[:, :, :-1]

        log_returns = X_sample[..., ::2]
        volatility = self.activation_fn(X_sample[..., 1::2])

        final_sample = torch.empty_like(X_sample)
        final_sample[..., ::2] = log_returns
        final_sample[..., 1::2] = volatility

        return final_sample

    def relu(self, x):
        """
        ReLU activation function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying ReLU.
        """
        return torch.clamp(x, min=0)

    def softplus(self, x):
        """
        Softplus activation function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying Softplus.
        """
        return torch.where(x > 0, x + torch.log1p(torch.exp(-x)),
                           torch.log1p(torch.exp(x)))


def init_generator(pretrain = False, **kwargs):
    print("Initialisation of the model.")
    config_dir = './config_ff.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    # config = {
    # "batch_size": 256,
    # "display_step": 500,
    # "epochs": 1000,
    # "hidden_dim": 256,
    # "input_dim": 11,
    # "lr": 0.003,
    # "n_flags": 5,
    # "n_flows": 28,
    # "n_lags": 5,
    # "output_dim": 10,
    # "vol_activation": "softplus"}
    generator_regular = FourierFlow(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden=config.hidden_dim,
        n_flows=config.n_flows,
        n_lags=config.n_lags,
        vol_activation=config.vol_activation,
        FFT=True,
        flip=True,
        normalize=True
    )
    generator_crisis = FourierFlow(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden=config.hidden_dim,
        n_flows=config.n_flows,
        n_lags=config.n_lags,
        vol_activation=config.vol_activation,
        FFT=True,
        flip=True,
        normalize=True
    )
    # If pretrain, loading data and pretrain model 
    if pretrain:
        print("Start loading data")
        with open("./ref_data.pkl", "rb") as f:
            loaded_array = pickle.load(f)

        training_data = torch.tensor(loaded_array)

        with open("./ref_label.pkl", "rb") as f:
            loaded_array = pickle.load(f)

        training_label = torch.tensor(loaded_array)

        regular_data = training_data[(training_label==0).squeeze()]
        crisis_data = training_data[(training_label==1).squeeze()]
        print("Training session")
        generator_regular.fit(regular_data.cpu().detach().numpy(), epochs=config.epochs,
                            batch_size=config.batch_size, learning_rate=config.lr,
                            display_step=config.display_step)

        generator_crisis.fit(crisis_data.cpu().detach().numpy(), epochs=config.epochs,
                            batch_size=config.batch_size, learning_rate=config.lr,
                            display_step=config.display_step)
        save_combined_model_dict(generator_regular, generator_crisis, './model_dict.pkl')
        generator_regular.eval()
        generator_crisis.eval()
        print("Finished Training")
    else:
        print("Loading the model.")
        PATH_TO_MODEL = './model_dict.pkl'
        with open(PATH_TO_MODEL, "rb") as f:
            combined_state_dict = pickle.load(f)

        generator_regular.load_state_dict(combined_state_dict['generator_regular'])
        generator_crisis.load_state_dict(combined_state_dict['generator_crisis'])

        generator_regular.eval()
        generator_crisis.eval()

    return generator_regular, generator_crisis


def save_combined_model_dict(generator_regular, generator_crisis, path):
    combined_state_dict = {
        'generator_regular': generator_regular.state_dict(),
        'generator_crisis': generator_crisis.state_dict()
    }
    with open(path, 'wb') as f:
        pickle.dump(combined_state_dict, f)
    print(f"Combined model checkpoint saved to {path}")