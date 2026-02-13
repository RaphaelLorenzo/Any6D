import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2
from estimater import Any6D
import trimesh
import argparse
import os
import yaml
import numpy as np
import time
device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).

# # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
# input_image = cv2.cvtColor(cv2.imread("PATH_TO_IMAGE.jpg"), cv2.COLOR_BGR2RGB)                       
# input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# # Infer 
# output = model.infer(input_image)
# """
# `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
# The maps are in the same size as the input image. 
# {
#     "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
#     "depth": (H, W),        # depth map
#     "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
#     "mask": (H, W),         # a binary mask for valid pixels. 
#     "intrinsics": (3, 3),   # normalized camera intrinsics
# }
# """

def main(args):
    
    save_path = f"./results_folder/{args.object_name}"
    out_meshes_path = f"./results_folder/{args.object_name}/meshes"
    out_poses_path = f"./results_folder/{args.object_name}/poses"
    out_points_path = f"./results_folder/{args.object_name}/points"
    out_depths_path = f"./results_folder/{args.object_name}/depths"
    out_images_path = f"./results_folder/{args.object_name}/images"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(out_meshes_path, exist_ok=True)
    os.makedirs(out_poses_path, exist_ok=True)
    os.makedirs(out_points_path, exist_ok=True)
    os.makedirs(out_depths_path, exist_ok=True)
    os.makedirs(out_images_path, exist_ok=True)
    
    
    images_path = os.path.join(args.dir_path, "images_jpg")
    images_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(images_path)) for f in fn if f.endswith(".jpg")]
    images_list.sort()
    masks_path = os.path.join(args.dir_path, "output_masks")
    masks_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(masks_path)) for f in fn if f.endswith(".png")]
    masks_list.sort()
    assert len(images_list) == len(masks_list)
    
    
    object_name = args.object_name
    mesh_path = os.path.join(args.dir_path, f"{object_name}.obj")
    assert os.path.exists(mesh_path)
    mesh = trimesh.load(mesh_path)
    
    resize_factor = 2
    
    calibration_path = os.path.join(args.dir_path, "calibration.yml")
    # camera info
    with open(calibration_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    intrinsic = np.array([[data["depth"]["fx"] / resize_factor, 0.0, data["depth"]["ppx"] / resize_factor], [0.0, data["depth"]["fy"] / resize_factor, data["depth"]["ppy"] / resize_factor], [0.0, 0.0, 1.0], ], )
    np.savetxt(os.path.join(save_path, f'K.txt'), intrinsic)    

    # initialize the estimator
    if args.only_depth:
        est = None
    else:   
        est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=out_meshes_path, debug=2)
        
    if args.no_depth:   
        depth_model = None
    else:
        depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             
        
    for it, (image_path, mask_path) in enumerate(zip(images_list, masks_list)):
        
        if args.skip > 0 and it%args.skip != 0:
            print(f"SKIPPING IMAGE {it+1}/{len(images_list)}")
            continue
        
        image_rn = os.path.basename(image_path).split(".")[0]
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)                       
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (image.shape[1] // resize_factor, image.shape[0] // resize_factor))
        mask = cv2.resize(mask, (mask.shape[1] // resize_factor, mask.shape[0] // resize_factor))
        
        mask = mask > 0

        if args.no_depth:
            depth = np.load(os.path.join(out_depths_path, f'{image_rn}.npy'))
            points = np.load(os.path.join(out_points_path, f'{image_rn}.npy'))
            assert(depth.shape == mask.shape)
            assert(depth.shape == image.shape[:2])
        else:
            tic = time.time()
            moge_image_input = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
            output_moge = depth_model.infer(moge_image_input)
            depth = output_moge['depth'].cpu().numpy() # (H, W)
            points = output_moge['points'].cpu().numpy() # (H, W, 3)
            toc = time.time()
            print(f"DEPTH ESTIMATION TIME (IMAGE {it+1}/{len(images_list)}): {toc - tic} seconds")
            
            if args.only_depth:
                continue
            
            # save depth and points as npy files
            np.save(os.path.join(out_depths_path, f'{image_rn}.npy'), depth)
            np.save(os.path.join(out_points_path, f'{image_rn}.npy'), points)
            # save the image for the colors
            cv2.imwrite(os.path.join(out_images_path, f'{image_rn}.jpg'), image)
            
            # import matplotlib.pyplot as plt
            # plt.subplot(1, 3, 1)
            # plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
            # plt.subplot(1, 3, 2)
            # plt.imshow(mask)
            # plt.subplot(1, 3, 3)
            # plt.imshow(depth)
            # plt.colorbar()
            # plt.show()
        
        print(image.shape, mask.shape, depth.shape)
        
        tic = time.time()
        print(image.shape, depth.shape, mask.shape)
        print(image.dtype, depth.dtype, mask.dtype)
        pred_pose = est.register_any6d(K=intrinsic, rgb=image, depth=depth, ob_mask=mask, iteration=5, name=image_rn, refinement=True, scaling_allow="none") # will save mesh in out_meshes_path/refine_init_mesh_{image_rn}.obj
        toc = time.time()
        print(f"POSE ESTIMATION TIME (IMAGE {it+1}/{len(images_list)}): {toc - tic} seconds")
        
        np.savetxt(os.path.join(out_poses_path, f'{image_rn}.txt'), pred_pose)

        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--dir_path", type=str, default="./demo_folder", help="Path to the directory containing the images")
    parser.add_argument("--object_name", type=str, default="bottle", help="Object name")
    parser.add_argument("--only_depth", action="store_true", default=False, help="Only estimate depth")
    parser.add_argument("--no_depth", action="store_true", default=False, help="No depth estimation")
    parser.add_argument("--skip", type=int, default=0, help="Skip every n images")
    args = parser.parse_args()
    
    main(args)