from django.shortcuts import render
from django.http import request

import os
import random
import argparse
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from .gan_module import Generator


def get_opt():
    module_dir = os.path.dirname(__file__)  
    dataroot = os.path.join(module_dir, 'data')
    
    args = {
        "image_dir": dataroot,
      
    }


    opt = argparse.Namespace(**args)
    return opt

    


def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    name = opt.name
    save_dir = os.path.join(opt.result_dir, name, opt.datamode)
    
    # Check if the "result" folder exists
    if os.path.exists(save_dir):
        # If it exists, remove it and all its contents
        # Use caution with this operation as it will permanently delete the folder and its contents
        # Make sure to use the correct path and double-check before proceeding
        try:
            # Remove the directory and its contents
            os.system(f'rmdir /s /q "{save_dir}"')  # For Windows
            # For Linux or macOS, use the following line instead:
            # os.system(f'rm -rf "{save_dir}"')
            print("Existing 'result' folder and its contents deleted.")
        except Exception as e:
            print("Error deleting the 'result' folder:", str(e))

    # Create a new "result" folder
    try:
        os.makedirs(save_dir)
        print("New 'result' folder created.")
    except Exception as e:
        print("Error creating the 'result' folder:", str(e))


    

    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)

    image_dir = os.path.join(save_dir, 'image')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    cloth_dir = os.path.join(save_dir, 'cloth')
    if not os.path.exists(cloth_dir):
        os.makedirs(cloth_dir)


    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        print(im_names, "  ", type(im_names))
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        im_names = []
        im_names.append("latest.jpg")
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        save_images(im, im_names, image_dir)
        save_images(c, im_names, cloth_dir)

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)



@torch.no_grad()
def main():
    
    args = get_opt()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nr_images = len(image_paths) if len(image_paths) < 6 else 6
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    random.shuffle(image_paths)
    for i in range(nr_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    # plt.show()
    plt.savefig("mygraph.png")


def index(request):
    print("fds")
    context = {'condition_met': False}
    if(request.method == "POST"):
        module_dir = os.path.dirname(__file__)  
        
        file1 = request.FILES['sentFile1']
        print(file1.name)

        # uploading.............
        upload_path = os.path.join(module_dir, 'static/data/')
        
        file_path = upload_path + "latest.jpg"
        with open(file_path, 'wb') as destination:
            for chunk in file1.chunks():
                destination.write(chunk)
        ###################################

        model = Generator(ngf=32, n_residual_blocks=9)

        model_file_path = os.path.join(module_dir, 'pretrained_model/state_dict.pth')
        ckpt = torch.load(model_file_path, map_location='cpu')

        model.load_state_dict(ckpt)
        model.eval()
        trans = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        fig, ax = plt.subplots(2, 1, figsize=(40, 20))

        img = Image.open(file_path).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0
        ax[0].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1].imshow(aged_face)
        plt.savefig("aging_gan/static/result/output_img.png")



        print(file1.name)
        
        
        context = {'condition_met': True}

    
    return render(request, 'home.html', context)