import utils.utils as utils

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import yaml


def build_loss(
    neural_net,
    optimizing_img,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config,
):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[
        content_feature_maps_index
    ].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction="mean")(
        target_content_representation, current_content_representation
    )

    style_loss = 0.0
    current_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(current_set_of_feature_maps)
        if cnt in style_feature_maps_indices
    ]
    for gram_gt, gram_hat in zip(
        target_style_representation, current_style_representation
    ):
        style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = (
        config["content_weight"] * content_loss
        + config["style_weight"] * style_loss
        + config["tv_weight"] * tv_loss
    )

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(
    neural_net,
    optimizer,
    target_representations,
    content_feature_maps_index,
    style_feature_maps_indices,
    config,
):
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(
            neural_net,
            optimizing_img,
            target_representations,
            content_feature_maps_index,
            style_feature_maps_indices,
            config,
        )
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step


def neural_style_transfer(config):
    content_img_path = os.path.join(
        config["content_images_dir"], config["content_img_name"]
    )
    style_img_path = os.path.join(config["style_images_dir"], config["style_img_name"])

    out_folder = (
        config["content_img_name"].split(".")[0]
        + "_"
        + config["style_img_name"].split(".")[0]
    )
    save_dir = os.path.join(config["output_img_dir"], out_folder)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config["img_hw"], device)
    style_img = utils.prepare_img(style_img_path, config["img_hw"], device)

    if config["init_method"] == "random":
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(
            loc=0, scale=90.0, size=content_img.shape
        ).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config["init_method"] == "content":
        init_img = content_img
    else:
        init_img = utils.prepare_img(
            style_img_path, np.asarray(content_img.shape[2:]), device
        )

    optimizing_img = Variable(init_img, requires_grad=True)
    (
        neural_net,
        content_feature_maps_index_name,
        style_feature_maps_indices_names,
    ) = utils.prepare_model(config["model"], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[
        content_feature_maps_index_name[0]
    ].squeeze(axis=0)
    target_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(style_img_set_of_feature_maps)
        if cnt in style_feature_maps_indices_names[0]
    ]
    target_representations = [
        target_content_representation,
        target_style_representation,
    ]

    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    if config["optimizer"] == "adam":
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(
            neural_net,
            optimizer,
            target_representations,
            content_feature_maps_index_name[0],
            style_feature_maps_indices_names[0],
            config,
        )
        for cnt in range(num_of_iterations[config["optimizer"]]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(
                    f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                )
                utils.save_result(optimizing_img, save_dir, config["saving_freq"], cnt)
    elif config["optimizer"] == "lbfgs":
        optimizer = LBFGS(
            (optimizing_img,),
            max_iter=num_of_iterations["lbfgs"],
            line_search_fn="strong_wolfe",
        )
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(
                neural_net,
                optimizing_img,
                target_representations,
                content_feature_maps_index_name[0],
                style_feature_maps_indices_names[0],
                config,
            )
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(
                    f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                )
                utils.save_result(optimizing_img, save_dir, config["saving_freq"], cnt)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return save_dir


if __name__ == "__main__":
    with open("configs/neural_style_transfer.yaml", "r") as yml:
        cfg = yaml.safe_load(yml)

    neural_style_transfer(cfg)
