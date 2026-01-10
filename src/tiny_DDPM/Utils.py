from matplotlib import pyplot as plt
import torch


def image_normalize(image):
    """
    Normalize a tensor image to the range [0, 1].

    Args:
        image (torch.Tensor): The input image tensor with shape [C, H, W].

    Returns:
        torch.Tensor: The normalized image tensor with shape [H, W, C].
    """
    image = image.cpu()
    n_channels = image.shape[0]
    for channel in range(n_channels):
        max_value = torch.max(image[channel])
        min_value = torch.min(image[channel])
        image[channel] = (image[channel] - min_value) / (max_value - min_value)

    image = image.permute(1, 2, 0)

    return image


def print_image(image):
    """
    Display a single tensor image.

    Args:
        image (torch.Tensor): The input image tensor.
    """
    image = image_normalize(image)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.show()


def print_2images(image1, image2):
    """
    Display two tensor images side by side.

    Args:
        image1 (torch.Tensor): The first image tensor.
        image2 (torch.Tensor): The second image tensor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_normalize(image1))
    axes[0].set_title("Image 1")

    axes[1].imshow(image_normalize(image2))
    axes[1].set_title("Image 2")

    plt.tight_layout()
    plt.show()


def print_digits(result):
    """
    Display a batch of digit images.

    Args:
        result (torch.Tensor): A batch of image tensors.
    """
    fig, axes = plt.subplots(1, 10, figsize=(10, 5))

    B = result.shape[0]
    for i in range(B):
        axes[i].imshow(image_normalize(result[i]))
        axes[i].set_title(i)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def print_result(result):
    """
    Print the results of the diffusion model, showing original and denoised images.

    Args:
        result (list): A list of tuples, where each tuple contains
                       (original_image, noised_image, denoised_image).
    """
    for original_image, noised_image, denoised_image in result:
        batch_size = original_image.shape[0]
        for idx in range(batch_size):
            print_2images(original_image[idx], denoised_image[idx])
            # print_image(image[idx])
            # print_image(noised_image[idx])
            # print_image(denoised_image[idx])


def print_seq(
    loss_values,
    label="Training Loss",
    x_label="Epoch",
    y_label="Loss",
    title="Loss vs. Epochs",
):
    """
    Plot a sequence of values, such as training loss over epochs.

    Args:
        loss_values (list or torch.Tensor): The values to plot.
        label (str): The label for the plot line.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
    """
    epochs = list(range(1, len(loss_values) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, "b-o", label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
