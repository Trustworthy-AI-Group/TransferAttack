import torch
import torch.nn.functional as F

from ...utils import *
from ...gradient.mifgsm import MIFGSM

import scipy.stats as st
import numpy as np

import os
import math


class ODI(MIFGSM):
    """
    ODI Attack
    'Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input (CVPR 2022)'(https://arxiv.org/pdf/2203.09123.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        kernel_type (str): the type of kernel (gaussian/uniform/linear).
        kernel_size (int): the size of kernel.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=2/255, epoch=300, decay=1., kernel_type='gaussian', kernel_size=15

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/odi/resnet18 --attack odi --model=resnet18 --targeted
    """

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., kernel_type='gaussian', kernel_size=5, targeted=False,
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='ODI', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.kernel = self.generate_kernel(kernel_type, kernel_size)
        self.config_idx = 101
        self.count = 0
        # self.renderer= Render3D(config_idx=self.config_idx,count=self.count)
        self.prob = 0.7

    def generate_kernel(self, kernel_type, kernel_size, nsig=3):
        """
        Generate the gaussian/uniform/linear kernel

        Arguments:
            kernel_type (str): the method for initilizing the kernel
            kernel_size (int): the size of kernel
        """
        if kernel_type.lower() == 'gaussian':
            x = np.linspace(-nsig, nsig, kernel_size)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        elif kernel_type.lower() == 'uniform':
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        elif kernel_type.lower() == 'linear':
            kern1d = 1 - np.abs(np.linspace((-kernel_size+1)//2, (kernel_size-1)//2, kernel_size)/(kernel_size**2))
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
        else:
            raise Exception("Unspported kernel type {}".format(kernel_type))

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def get_grad(self, loss, delta, **kwargs):
        """
        Overridden for TIM attack.
        """
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        grad = F.conv2d(grad, self.kernel, stride=1, padding='same', groups=3)
        return grad

    def get_loss(self, logits, label):
        real = logits.gather(1,label.unsqueeze(1)).squeeze(1)
        logit_dists = ( -1 * real)
        loss = logit_dists.sum()
        return -loss

    def transform(self, data, renderer,**kwargs):
        c = np.random.rand(1)
        if c <= self.prob:
            x_ri=data.clone()
            for i in range(data.shape[0]):
                x_ri[i]=renderer.render(data[i].unsqueeze(0), self.device)
            return  x_ri
        else:
            return  data

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        renderer = Render3D(config_idx=self.config_idx, count=self.count, device=self.device)
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, renderer))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        # torch.cuda.empty_cache()
        return delta.detach()


exp_configuration={
    101:{ # BEST MODEL / WE PICK THIS AS MAIN MODEL
        'p':1.,  # "prob for DI and RE"
        'alpha':2,
        'max_iterations':300, # "max_iterations"
        'num_images':1000,
        'source_model_names':['ResNet50','vgg16', 'DenseNet121', 'inception_v3'],
        'target_model_names':['vgg16','ResNet18', 'ResNet50', 'DenseNet121', 'inception_v3', 'inception_v4_timm', 'mobilenet_v2','inception_resnet_v2',  'adv_inception_v3', 'ens_adv_inception_resnet_v2'],
        'attack_methods': {'ODI-MI-TI': 'OTM3'},
        'number_of_v_samples':5,
        'number_of_si_scales':5,
        'shininess':0.5,
        'source_3d_models':['pack','pillow','book'],
        # 'source_3d_models':['pillow'],
        'rand_elev':(-35,35),
        'rand_azim':(-35,35),
        'rand_angle':(-35,35),
        'min_dist':0.8, 'rand_dist':0.4,
        'light_location':[0.0, 0.0,4.0],
        'rand_light_location':4,
        'rand_ambient_color':0.3,
        'ambient_color':0.6,
        'rand_diffuse_color':0.5,
        'diffuse_color':0.0,
        'specular_color':0.0,
        'background_type':'random_pixel',
        'texture_type':'random_solid',
        'visualize':False,
        'comment':'3 model ensemble'
    }
}


class Render3D(object):
    def __init__(self, config_idx=1, count=1, device=None):
        exp_settings=exp_configuration[config_idx] # Load experiment configuration

        self.config_idx=config_idx
        self.count=count
        self.eval_count=0
        self.device = device

        ## Pytorch3D ########################################
        # Util function for loading meshes
        from pytorch3d.io import load_objs_as_meshes, load_obj

        # Data structures and functions for rendering
        from pytorch3d.structures import Meshes
        # from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
        # from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
        from pytorch3d.renderer import (
            look_at_view_transform,
            FoVPerspectiveCameras,
            PointLights,
            # DirectionalLights,
            # look_at_rotation,
            Materials,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader,
            # TexturesUV,
            # TexturesVertex,
            blending
        )

        raster_settings = RasterizationSettings(
            image_size=224,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Just initialization. light position and brightness are randomly set for each inference
        self.lights = PointLights(device=self.device, ambient_color=((0.3, 0.3, 0.3),), diffuse_color=((0.5, 0.5, 0.5), ), specular_color=((0.5, 0.5, 0.5), ),
        location=[[0.0, 3.0,0.0]])

        R, T = look_at_view_transform(dist=1.0, elev=0, azim=0)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        self.materials = Materials(
            device=self.device,
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=exp_settings['shininess']
        )

        # Note: the background color of rendered images is set to -1 for proper blending
        blend_params = blending.BlendParams(background_color=[-1., -1., -1.])

        # Create a renderer by composing a mesh rasterizer and a shader.
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=self.lights,
                blend_params=blend_params
            )
        )
        # 3D Model setting
        # {'3d model name', ['filename', x, y, w, h, initial distance, initial elevation, initial azimuth, initial translation]}
        self.model_settings={'pack':['pack.obj',255,255,510,510,1.2,0,0,[0,0.02,0.]],
        'cup':['cup.obj',693,108,260,260,1.7,0,0,[0.,-0.1,0.]],
        'pillow':['pillow.obj',10,10,470,470,1.7,0,0],
        't_shirt':['t_shirt_lowpoly.obj',180,194,240,240,1.2,0,0,[0.0,0.05,0]],
        'book':['book.obj',715,66,510,510,1.3,0,0,[0.3,0.,0]],
        '1ball':['1ball.obj',359,84,328,328,2.1,-40,-10],
        '2ball':['2ball.obj',359,84,328,328,1.9,-40,-10,[-0.1,0.,0]],
        '3ball':['3ball.obj',359,84,328,328,1.8,-25,-10,[-0.1,0.15,0]],
        '4ball':['4ball.obj',359,84,328,328,1.8,-25,-10,[0.,0.1,0]]
        }

        self.source_models=exp_settings['source_3d_models'] # Import source model list

        self.background_img=torch.zeros((1,3,224,224)).to(self.device)

        for src_model in self.source_models:
            self.model_settings[src_model][0]=self.load_object(self.model_settings[src_model][0], self.device)

        # The following code snippet is for 'blurred image' backgrounds.
        kernel_size=50
        kernel = self.gkern(kernel_size, 15).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        self.gaussian_kernel = torch.from_numpy(gaussian_kernel).to(self.device)

    def gkern(self, kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def render(self, img, device):
        from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
        self.eval_count+=1

        exp_settings=exp_configuration[self.config_idx]

        # Default experimental settings.
        if 'background_type' not in exp_settings:
            exp_settings['background_type']='none'
        if 'texture_type' not in exp_settings:
            exp_settings['texture_type']='none'
        if 'visualize' not in exp_settings:
            exp_settings['visualize']=False

        x_adv=img
        # Randomly select an object from the source object pool
        pick_idx=np.random.randint(low=0,high=len(self.source_models))

        # Load the 3D mesh
        mesh=self.model_settings[self.source_models[pick_idx]][0]

        # Load the texture map
        texture_image=mesh.textures.maps_padded()

        texture_type=exp_settings['texture_type']

        if texture_type=='random_pixel':
            texture_image.data=torch.rand_like(texture_image, device=self.device)
        elif texture_type=='random_solid': # Default setting
            texture_image.data=torch.ones_like(texture_image, device=self.device)*(torch.rand((1,1,1,3), device=self.device)*0.6+0.1)
        elif  texture_type=='custom':
            texture_image.data=torch.ones_like(texture_image, device=self.device)*torch.FloatTensor( [ 0/255.,0./255.,0./255.]).view((1,1,1,3)).to(self.device)

        (pattern_h,pattern_w)=(self.model_settings[self.source_models[pick_idx]][4],self.model_settings[self.source_models[pick_idx]][3])

        # Resize the input image
        resized_x_adv=F.interpolate(x_adv, size=(pattern_h, pattern_w), mode='bilinear').permute(0,2,3,1)
        # Insert the resized image into the canvas area of the texture map
        (x,y)=self.model_settings[self.source_models[pick_idx]][1],self.model_settings[self.source_models[pick_idx]][2]
        texture_image[:,y:y+pattern_h,x:x+pattern_w,:]=resized_x_adv

        # Adjust the light parameters
        self.lights.location = torch.tensor(exp_settings['light_location'], device=self.device)[None]+(torch.rand((3,), device=self.device)*exp_settings['rand_light_location']-exp_settings['rand_light_location']/2)
        self.lights.ambient_color=torch.tensor([exp_settings['ambient_color']]*3, device=self.device)[None]+(torch.rand((1,),device=self.device)*exp_settings['rand_ambient_color'])
        self.lights.diffuse_color=torch.tensor([exp_settings['diffuse_color']]*3, device=self.device)[None]+(torch.rand((1,),device=self.device)*exp_settings['rand_diffuse_color'])
        self.lights.specular_color=torch.tensor([exp_settings['specular_color']]*3, device=self.device)[None]


        # Adjust the camera parameters
        rand_elev=torch.randint(exp_settings['rand_elev'][0],exp_settings['rand_elev'][1]+1, (1,))
        rand_azim=torch.randint(exp_settings['rand_azim'][0],exp_settings['rand_azim'][1]+1, (1,))
        rand_dist=(torch.rand((1,))*exp_settings['rand_dist']+exp_settings['min_dist'])
        rand_angle=torch.randint(exp_settings['rand_angle'][0],exp_settings['rand_angle'][1]+1, (1,))



        R, T = look_at_view_transform(dist=(self.model_settings[self.source_models[pick_idx]][5])*rand_dist, elev=self.model_settings[self.source_models[pick_idx]][6]+rand_elev,
        azim=self.model_settings[self.source_models[pick_idx]][7]+rand_azim,up=((0,1,0),))

        if len(self.model_settings[self.source_models[pick_idx]])>8: # Apply initial translation if it is given.
            TT=T+torch.FloatTensor(self.model_settings[self.source_models[pick_idx]][8])
        else:
            TT=T

        # Compute rotation matrix for tilt
        angles=torch.FloatTensor([[0,0,rand_angle*math.pi/180]]).to(self.device)
        rot=self.compute_rotation(angles, device).squeeze()
        R=R.to(self.device)

        R=torch.matmul(rot,R)

        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=TT)

        # Render the mesh with the modified rendering environments.
        rendered_img = self.renderer(mesh, lights=self.lights, materials=self.materials, cameras=self.cameras)

        rendered_img=rendered_img[:, :, :,:3] # RGBA -> RGB

        rendered_img=rendered_img.permute(0,3,1,2) # B X H X W X C -> B X C X H X W

        background_type=exp_settings['background_type']

        # The following code snippet is for blending
        rendered_img_mask = 1.-(rendered_img.sum(dim=1,keepdim=True)==-3.).float()
        rendered_img = torch.clamp(rendered_img, 0., 1.)
        if background_type=='random_pixel':
            background_img=torch.rand_like(rendered_img,device=self.device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='random_solid':
            background_img=torch.ones_like(rendered_img,device=self.evice)*torch.rand((1,3,1,1),device=self.device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='blurred_image':
            background_img=img.clone().detach()
            background_img = F.conv2d(background_img, self.gaussian_kernel, bias=None, stride=1, padding='same', groups=3)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='custom':
            background_img=torch.ones_like(rendered_img,device=self.device)*torch.FloatTensor( [ 0/255.,0./255.,0./255.]).view((1,3,1,1)).to(self.device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        else:
            result_img=rendered_img

        import cv2
        if exp_settings['visualize']==True:
            result_img_npy=result_img.permute(0,2,3,1)
            result_img_npy=result_img_npy.squeeze().cpu().detach().numpy()
            converted_img=cv2.cvtColor(result_img_npy, cv2.COLOR_BGR2RGB)
            cv2.imshow('Video', converted_img) #[0, ..., :3]
            key=cv2.waitKey(1) & 0xFF

        return result_img
    
    def compute_rotation(self, angles, device):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(device)
        zeros = torch.zeros([batch_size, 1]).to(device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def rigid_transform(self, vs, rot, trans):
        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)
        return vs_t

    def load_object(self, obj_file_name, device):
        from pytorch3d.structures import Meshes
        from pytorch3d.io import load_objs_as_meshes, load_obj
        obj_filename = os.path.join("./transferattack/input_transformation/odi/obj/", obj_file_name)

        # Load the 3D model using load_obj
        verts, faces, aux = load_obj(obj_filename)

        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)

        # We scale normalize and center the mesh.
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        angles=torch.FloatTensor([[90*math.pi/180,0,0]]).to(device)

        rot=self.compute_rotation(angles, device).squeeze()

        verts=torch.matmul(verts,rot)

        # Get the scale normalized textured mesh
        mesh = load_objs_as_meshes([obj_filename], device=device)
        mesh = Meshes(verts=[verts], faces=[faces_idx],textures=mesh.textures)
        return mesh