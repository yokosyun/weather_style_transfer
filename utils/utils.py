import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os

from models.definitions.vgg_nets import Vgg16, Vgg19


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def load_image(img_path, target_shape):
    if not os.path.exists(img_path):
        raise Exception(f"Path does not exist: {img_path}")
    img = cv.imread(img_path)[:, :, ::-1]

    img = cv.resize(
        img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC
    )

    img = img.astype(np.float32)
    img /= 255.0
    return img


def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL),
        ]
    )

    img = transform(img).to(device).unsqueeze(0)

    return img


def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])


def save_result(
    optimizing_img,
    dump_path,
    saving_freq,
    img_id,
):
    out_img = optimizing_img.squeeze(axis=0).to("cpu").detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    img_id += 1

    if saving_freq > 0 and (img_id % saving_freq == 0):
        out_img_name = str(img_id).zfill(4) + ".jpg"
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype("uint8")
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f"Expected numpy array got {type(x)}")


def prepare_model(model_name):
    if model_name == "vgg16":
        model = Vgg16(requires_grad=False, show_progress=True)
    elif model_name == "vgg19":
        model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise ValueError(f"{model} not supported.")

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    content_fms_index_name = (
        content_feature_maps_index,
        layer_names[content_feature_maps_index],
    )

    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(
        torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    )
