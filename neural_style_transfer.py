import utils.utils as utils

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import yaml


def build_loss(
    model,
    optimizing_img,
    target_content_feats,
    target_style_feats,
    content_feat_idx,
    style_feat_idxs,
    loss_weights,
):
    current_feat_maps = model(optimizing_img)

    current_content_feats = current_feat_maps[content_feat_idx].squeeze(axis=0)

    content_loss = torch.nn.MSELoss(reduction="mean")(
        target_content_feats, current_content_feats
    )

    style_loss = 0.0
    current_style_feats = [
        utils.gram_matrix(x)
        for idx, x in enumerate(current_feat_maps)
        if idx in style_feat_idxs
    ]

    for gram_gts, gram_hats in zip(target_style_feats, current_style_feats):
        for gram_gt in gram_gts:
            style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt, gram_hats[0])
    style_loss /= len(gram_gts)
    style_loss /= len(target_style_feats)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = (
        loss_weights["content_weight"] * content_loss
        + loss_weights["style_weight"] * style_loss
        + loss_weights["tv_weight"] * tv_loss
    )

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(
    model,
    optimizer,
    target_content_feats,
    target_style_feats,
    content_feat_idx,
    style_feat_idxs,
    loss_weights,
):
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(
            model,
            optimizing_img,
            target_content_feats,
            target_style_feats,
            content_feat_idx,
            style_feat_idxs,
            loss_weights,
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

    style_img_names = os.path.join(*config["style_img_names"])
    style_img_names = style_img_names.replace(".jpg", "").replace("/", "-")

    out_folder = config["content_img_name"].split(".")[0] + "-" + style_img_names
    save_dir = os.path.join(config["save_dir"], out_folder)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config["img_hw"], device)

    style_imgs = []
    for img_name in config["style_img_names"]:
        style_img_path = os.path.join(config["style_images_dir"], img_name)
        style_img = utils.prepare_img(style_img_path, config["img_hw"], device)
        style_imgs.append(style_img)
    style_img = torch.cat(style_imgs, dim=0)

    optimizing_img = Variable(content_img.clone(), requires_grad=True)

    model = utils.prepare_model(config["model"])
    model.eval()
    model = model.to(device)

    content_feat_maps = model(content_img)
    style_feat_maps = model(style_img)

    target_content_feats = content_feat_maps[model.content_feat_idx].squeeze(axis=0)

    target_style_feats = [
        utils.gram_matrix(x)
        for idx, x in enumerate(style_feat_maps)
        if idx in model.style_feat_idxs
    ]

    if config["optimizer"] == "adam":
        optimizer = Adam((optimizing_img,), **config["Adam"])
        tuning_step = make_tuning_step(
            model,
            optimizer,
            target_content_feats,
            target_style_feats,
            model.content_feat_idx,
            model.style_feat_idxs,
            config["loss_weights"],
        )
        for iter in range(config["num_of_iterations"]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(
                    f"Adam | iteration: {iter:03}, total loss={total_loss.item():12.4f}, content_loss={content_loss.item():12.4f}, style loss={style_loss.item():12.4f}, tv loss={tv_loss.item():12.4f}"
                )
                utils.save_result(optimizing_img, save_dir, config["saving_freq"], iter)
    elif config["optimizer"] == "lbfgs":
        optimizer = LBFGS(
            (optimizing_img,),
            max_iter=config["num_of_iterations"],
            line_search_fn="strong_wolfe",
        )
        iter = 0

        def closure():
            nonlocal iter
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(
                model,
                optimizing_img,
                target_content_feats,
                target_style_feats,
                model.content_feat_idx,
                model.style_feat_idxs,
                config["loss_weights"],
            )
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(
                    f"L-BFGS | iteration: {iter:03}, total loss={total_loss.item():12.4f}, content_loss={content_loss.item():12.4f}, style loss={style_loss.item():12.4f}, tv loss={tv_loss.item():12.4f}"
                )
                utils.save_result(optimizing_img, save_dir, config["saving_freq"], iter)

            iter += 1
            return total_loss

        optimizer.step(closure)

    return save_dir


if __name__ == "__main__":
    with open("configs/neural_style_transfer.yaml", "r") as yml:
        cfg = yaml.safe_load(yml)

    neural_style_transfer(cfg)
