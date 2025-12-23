import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from MyDiffusion.modules.UNet import UNet
from MyDiffusion.forward_encoder import ForwardEncoder
from MyDiffusion.reverse_decoder import ReverseDecoder
from MyDiffusion.noise_sheduler import NoiseSchedule


class Diffusion:
    """
    Main class for the Diffusion model, handling training, evaluation, and model saving/loading.
    """

    def __init__(
        self,
        model,
        n_timesteps,
        train_set=None,
        test_set=None,
        train_batch_size=8,
        test_batch_size=8,
        learning_rate=0.0001,
        device=None,
    ):
        """
        Initializes the Diffusion model components.

        Args:
            model (nn.Module): The neural network model to be trained (e.g., UNet, DiT).
            n_timesteps (int): Number of timesteps for the diffusion process.
            train_set (Dataset, optional): Training dataset. Defaults to None.
            test_set (Dataset, optional): Testing dataset. Defaults to None.
            train_batch_size (int, optional): Batch size for training. Defaults to 8.
            test_batch_size (int, optional): Batch size for testing. Defaults to 8.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to None.
        """

        self.n_timesteps = n_timesteps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = device

        # Model
        self.model = model.to(device)

        # Noise Scheduler
        self.noise_schedule = NoiseSchedule(
            n_timesteps=n_timesteps,
            device=device,
            # init_type="exponential",
        )

        # Forward Encoder
        self.encoder = ForwardEncoder(noise_schedule=self.noise_schedule, device=device)

        # Reverse Decoder
        self.decoder = ReverseDecoder(
            noise_schedule=self.noise_schedule, g=self.model, device=device
        )

        # Optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # DataLoader
        if train_set:
            self.training_loader = DataLoader(
                train_set, batch_size=train_batch_size, shuffle=True
            )
        if test_set:
            self.testing_loader = DataLoader(
                test_set, batch_size=test_batch_size, shuffle=True
            )

    def save(self, path="./model.pt"):
        """Saves the model's state dictionary."""
        torch.save(self.model.state_dict(), path)

    def load(self, path="./model.pt"):
        """Loads the model's state dictionary."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def train_one_epoch(self, p_uncond=0.1, w=1):
        """
        Runs a single training epoch.

        Args:
            p_uncond (float, optional): Probability of unconditional training for classifier-free guidance. Defaults to 0.1.
            w (int, optional): Guidance weight. Defaults to 1.

        Returns:
            float: The average loss for the epoch.
        """

        running_loss = 0

        for i, data in enumerate(tqdm(self.training_loader)):

            # inputs = [B, 1, 32, 32]
            inputs, label = data
            inputs = inputs.to(self.device)
            # print(inputs.shape)

            batch_size = inputs.shape[0]

            # sampled timestep and conditional variables
            t = torch.randint(0, self.n_timesteps, (batch_size,)).to(self.device)
            c = label.to(self.device)

            # Forward Encoder (image -> noise)
            noised_image, epsilon = self.encoder.noise(inputs, t)

            # Classifier Free Guidance
            if random.random() < p_uncond:
                c = None
            outputs = self.model(noised_image, t, c)

            loss = self.criterion(outputs, epsilon)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.training_loader)

    def train(self, n_epoch=5, n_iter_limit=None, p_uncond=0.1):
        """
        Trains the diffusion model for a specified number of epochs.

        Args:
            n_epoch (int, optional): Number of epochs to train for. Defaults to 5.
            n_iter_limit (int, optional): Limit on the number of iterations per epoch. Defaults to None.
            p_uncond (float, optional): Probability of unconditional training. Defaults to 0.1.

        Returns:
            list: A list containing the average loss for each epoch.
        """

        history = []

        for epoch in range(n_epoch):
            self.model.train(True)
            print("EPOCH {}:".format(epoch + 1))
            avg_loss = self.train_one_epoch(p_uncond=p_uncond)
            history.append(avg_loss)
            print("# epoch {} avg_loss: {}".format(epoch + 1, avg_loss))

            # Create checkpoints directory if it doesn't exist
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model_filename = "{}_T{}_E{}.pt".format(
                self.model.__class__.__name__, self.n_timesteps, epoch + 1
            )
            model_path = checkpoint_dir / model_filename
            history_path = checkpoint_dir / "history.pt"

            torch.save(self.model.state_dict(), model_path)
            torch.save(torch.tensor(history), history_path)
            print(f"Model saved to: {model_path}")

        return history

    def evaluate(
        self,
        epochs=None,
        sampling_type="DDPM",
        sampling_timestep=10,
        custom_sampling_steps=None,
        w=0,
        handler=None,
    ):
        """
        Evaluates the model on the test set.

        Args:
            epochs (int, optional): Number of evaluation batches to run. Defaults to None (runs through the whole dataset).
            sampling_type (str, optional): Type of sampling ('DDPM' or 'DDIM'). Defaults to "DDPM".
            sampling_timestep (int, optional): Number of timesteps for DDIM sampling. Defaults to 10.
            custom_sampling_steps (list, optional): Custom timesteps for DDIM sampling. Defaults to None.
            w (int, optional): Guidance weight. Defaults to 0.
            handler (function, optional): A function to handle the output images. Defaults to None.

        Returns:
            list: A list of tuples containing (original_image, noised_image, denoised_image).
        """
        self.decoder.g = self.model
        result = []
        for i, data in enumerate(tqdm(self.testing_loader)):

            # inputs = [B, 1, 32, 32]
            inputs, label = data
            inputs = inputs.to(self.device)

            B = inputs.shape[0]

            # timestep
            t = torch.full((B,), self.n_timesteps - 1).to(self.device)
            c = label.to(self.device)

            # outputs = [B, 1, 28, 28]
            noised_image, epsilon = self.encoder.noise(inputs, t)

            # denoised image
            denoised_image = None
            if sampling_type == "DDPM":
                denoised_image = self.decoder.DDPM_sampling(
                    noised_image, self.n_timesteps, c=c, w=w
                )
            if sampling_type == "DDIM":
                denoised_image = self.decoder.DDIM_sampling(
                    noised_image,
                    self.n_timesteps,
                    c=c,
                    w=w,
                    sampling_steps=sampling_timestep,
                    custom_sampling_steps=custom_sampling_steps,
                )

            result.append((inputs, noised_image, denoised_image))
            if handler is not None:
                handler(inputs, noised_image, denoised_image)

            if epochs is not None and i == epochs - 1:
                break

        return result

    def evaluate_with_noise(
        self,
        epochs=10,
        sampling_type="DDPM",
        sampling_timestep=10,
        custom_sampling_steps=None,
        w=0,
        handler=None,
    ):
        """
        Generates images from random noise.

        Args:
            epochs (int, optional): Number of batches to generate. Defaults to 10.
            sampling_type (str, optional): Type of sampling ('DDPM' or 'DDIM'). Defaults to "DDPM".
            sampling_timestep (int, optional): Number of timesteps for DDIM sampling. Defaults to 10.
            custom_sampling_steps (list, optional): Custom timesteps for DDIM sampling. Defaults to None.
            w (int, optional): Guidance weight. Defaults to 0.
            handler (function, optional): A function to handle the output images. Defaults to None.

        Returns:
            list: A list of tuples containing (original_noise, noised_image, denoised_image).
        """
        self.decoder.g = self.model
        B = self.test_batch_size

        result = []

        for _ in range(epochs):

            # inputs = [B, 1, 32, 32]
            inputs = torch.randn((B, 1, 32, 32)).to(self.device)

            # Timestep and Conditional variables
            t = torch.full((B,), self.n_timesteps - 1).to(self.device)
            c = torch.randint(0, 10, (B,)).to(self.device)
            noised_image = inputs

            # Denoised Images
            denoised_image = None
            if sampling_type == "DDPM":
                denoised_image = self.decoder.DDPM_sampling(
                    noised_image, self.n_timesteps, c=c, w=w
                )
            if sampling_type == "DDIM":
                denoised_image = self.decoder.DDIM_sampling(
                    noised_image,
                    self.n_timesteps,
                    c=c,
                    w=w,
                    sampling_steps=sampling_timestep,
                    custom_sampling_steps=custom_sampling_steps,
                )

            if handler is not None:
                handler(inputs, noised_image, denoised_image)
            result.append((inputs, noised_image, denoised_image))

        return result
