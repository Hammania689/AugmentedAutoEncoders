import time
from pathlib import Path

import cv2
import gin
import torch
import wandb
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from src.datasets.render_wrapper import *

def get_abs_from_relative(gin_path)-> str:
    cur_path = Path(__file__).absolute().parent

    start_path = cur_path
    query_path = tuple()
    for p in Path(gin_path).parts:
        if p == '..':
            start_path = start_path.parent
        elif p == '.':
            continue
        else:
            query_path += (p,)

    query_path = "/".join(query_path)
    
    query_path = start_path / query_path
    return str(query_path)

np.random.seed(42)
seeded_colors = np.random.randint(0, 256, 3) / 255

class PoseEvaluator():

    def __init__(self,
                 aae,
                 codebook,
                 obj_list,
                 obj_idx,
                 cfg_list,
                 modality,
                 cad_model_dir,
                 obj_ctg="tless",
                 visualize=True,
                 gpu_id=0):

        self.__dict__.update(vars())

        self.target_obj_cfg = self.cfg_list[0]
        self.target_obj_idx = obj_idx
        # renderer
        self.intrinsics = np.array([[self.cfg_list[0].PF.FU, 0, self.cfg_list[0].PF.U0],
                                   [0, self.cfg_list[0].PF.FV, self.cfg_list[0].PF.V0],
                                   [0, 0, 1.]], dtype=np.float32)

        self.renderer = render_wrapper(self.obj_list, self.intrinsics, gpu_id=self.gpu_id,
                                       model_dir=cad_model_dir,
                                       model_ctg=self.obj_ctg,
                                       im_w=int(self.cfg_list[0].PF.W),
                                       im_h=int(self.cfg_list[0].PF.H),
                                       initialize_render=True)

        # evaluation module
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        # for logging
        self.log_dir = get_abs_from_relative('../../results')
        self.log_created = False
        self.log_pose = None
        self.log_error = None

        self.log_err_t = []
        self.log_err_tx = []
        self.log_err_ty = []
        self.log_err_tz = []

        self.log_err_r = []
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []

        self.log_max_sim = []
        self.recon_reel = []

        self.codebook_latent, self.codebook_poses = codebook

    @gin.configurable
    def log_video(self, img_list, log_to_wandb: str, wandb_entity: str):
        [cv2.imwrite(f"{self.log_dir}/{index:04d}.png", v) for index, v in enumerate(tqdm(img_list, desc="Writing images to disk"))]

        (Path(self.log_dir) / "recon").mkdir(exist_ok=True, parents=True)
        [save_image(r, f"{self.log_dir}/recon/{index:04d}.png", normalize=True) for index, r in enumerate(self.recon_reel)]

        offset = len(self.recon_reel) // 8
        reel_log = {}
        for index, start in enumerate(list(range(0, 503, offset))):
            cur_reel = make_grid(self.recon_reel[start:start+offset], normalize=True)
            save_image(cur_reel, f"{self.log_dir}/recon_reel_{index:04d}.png", normalize=True)
            reel_log.update(**{f"recon_reel_{index:04d}": wandb.Image(cur_reel)})
        
        if log_to_wandb:
            vid = np.stack(img_list)
            wandb.init(project="AugmentedAutoEncoders-test", entity=wandb_entity)
            wandb.log({"video": wandb.Video((vid).transpose(0, 3, 1, 2), fps=15, format="mp4"),
                       "gif":   wandb.Video((vid).transpose(0, 3, 1, 2), fps=5, format="gif"),
                       **reel_log})


    # initialize PoseRBPF
    def estimate_orientation(self, image, intrinsics):

        # crop the input image according to uv z
        images_roi_cuda, scale_roi = get_rois_cuda(image.detach(), self.gt_uv, self.gt_t[2].reshape(1, -1),
                                                         self.target_obj_cfg.PF.FU.reshape(1, -1),
                                                         self.target_obj_cfg.PF.FV.reshape(1, -1),
                                                         self.target_obj_cfg.TRAIN.RENDER_DIST[0])

        # just crop in the center
        roi_info = torch.zeros(images_roi_cuda.size(0), 5).float().cuda()
        roi_info[:, 0] = torch.arange(images_roi_cuda.size(0))
        roi_info[:, 1] = 128.0 - 128.0 / 2
        roi_info[:, 2] = 128.0 - 128.0 / 2
        roi_info[:, 3] = 128.0 + 128.0 / 2
        roi_info[:, 4] = 128.0 + 128.0 / 2

        # compute the codes
        recon, query_latent = self.aae.forward(images_roi_cuda)
        query_latent = query_latent.detach()
        recon = recon.detach()
        reel = torch.cat((images_roi_cuda.cpu(), recon.cpu()), dim=-1) * 255
        self.recon_reel.append(reel.squeeze())

        # pw_latent = pairwise_cosine_distances(query_latent, self.codebook_latent).squeeze()
        pw_latent = self.aae.pairwise_cosine_distances(query_latent, self.codebook_latent).squeeze()
        matching_idx = torch.argmax(pw_latent)
        self.cur_rgb_sim = pw_latent[matching_idx]
        self.rot_bar = quat2mat(self.codebook_poses[matching_idx][3:].cpu())

        # compute roi
        """
        pose = np.zeros((7,), dtype=np.float32)
        pose[4:] = self.trans_bar
        pose[:4] = mat2quat(self.rot_bar)
        points = self.points_list[self.target_obj_idx]
        # box = self.compute_box(pose, points)
        """

    
    def display_result(self, step, steps):
        qco = mat2quat(self.gt_rotm)

        filter_rot_error_bar = abs(single_orientation_error(qco, mat2quat(self.rot_bar)))

        trans_bar_f = self.trans_bar

        filter_trans_error_bar = np.linalg.norm(trans_bar_f - self.gt_t)

        self.log_err_t.append(filter_trans_error_bar)
        self.log_err_r.append(filter_rot_error_bar * 57.3)

        self.log_err_tx.append(np.abs(trans_bar_f[0] - self.gt_t[0]))
        self.log_err_ty.append(np.abs(trans_bar_f[1] - self.gt_t[1]))
        self.log_err_tz.append(np.abs(trans_bar_f[2] - self.gt_t[2]))

        rot_err_axis = single_orientation_error_axes(qco, mat2quat(self.rot_bar))
        self.log_err_rx.append(np.abs(rot_err_axis[0]))
        self.log_err_ry.append(np.abs(rot_err_axis[1]))
        self.log_err_rz.append(np.abs(rot_err_axis[2]))

        self.log_max_sim.append(self.cur_rgb_sim)

        print('     step {}/{}: translation error (filter)   = {:.4f} cm'.format(step+1, int(steps),
                                                                                filter_trans_error_bar * 100))
        print('     step {}/{}: RGB Similarity   = {:.3f}'.format(step + 1, int(steps), self.cur_rgb_sim))
        if self.modality == 'rgbd':
            print('     step {}/{}: Depth Similarity   = {:.3f}'.format(step + 1, int(steps), self.max_sim_depth))
        print('     step {}/{}: uvz error (filter)           = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                 self.uv_bar[0] - self.gt_uv[0],
                                                                                                 self.uv_bar[1] - self.gt_uv[1],
                                                                                                 (trans_bar_f[2] - self.gt_t[2]) * 100))
        print('     step {}/{}: xyz error (filter)           = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                self.log_err_tx[-1
                                                                                                ] * 1000,
                                                                                                self.log_err_ty[
                                                                                                    -1] * 1000,
                                                                                                self.log_err_tz[
                                                                                                    -1] * 1000))
        print('     step {}/{}: xyz rotation err (filter)    = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                self.log_err_rx[-1
                                                                                                ] * 57.3,
                                                                                                self.log_err_ry[
                                                                                                    -1] * 57.3,
                                                                                                self.log_err_rz[
                                                                                                    -1] * 57.3))
        print('     step {}/{}: rotation error (filter)      = {:.4f} deg'.format(step+1, int(steps),
                                                                                filter_rot_error_bar * 57.3))

        return filter_rot_error_bar * 57.3


    def save_log(self, sequence, filename, with_gt=True, tless=False):
        if not self.log_created:
            self.log_pose = open(self.log_dir + "/Pose_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence), "w+")
            if with_gt:
                self.log_pose_gt = open(self.log_dir + "/Pose_GT_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence),
                                     "w+")
                self.log_error = open(self.log_dir + "/Error_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence), "w+")
            self.log_created = True

        q_log = mat2quat(self.rot_bar)
        self.log_pose.write('{} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} \n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                                               filename[0],
                                                                                               self.trans_bar[0],
                                                                                               self.trans_bar[1],
                                                                                               self.trans_bar[2],
                                                                                               q_log[0],
                                                                                               q_log[1],
                                                                                               q_log[2],
                                                                                               q_log[3]))

        if with_gt:
            q_log_gt = mat2quat(self.gt_rotm)
            self.log_pose_gt.write(
                '{} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} \n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                                   filename[0],
                                                                                   self.gt_t[0],
                                                                                   self.gt_t[1],
                                                                                   self.gt_t[2],
                                                                                   q_log_gt[0],
                                                                                   q_log_gt[1],
                                                                                   q_log_gt[2],
                                                                                   q_log_gt[3]))
            self.log_error.write('{} {} {:.5f} {:.5f} \n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                filename[0],
                                                                self.log_err_t[-1] * 100,
                                                                self.log_err_r[-1] * 57.3))

        # for tless dataset, save the results as sixd challenge format
        if tless:
            obj_id_sixd = int(self.target_obj_cfg.PF.TRACK_OBJ[-2:])
            seq_id_sixd = int(filename[0][:2])
            img_id_sixd = int(filename[0][-4:])
            """
            save_folder_sixd = self.target_obj_cfg.PF.SAVE_DIR[
                               :-7] + 'dpf_tless_primesense_all/'  # .format(obj_id_sixd)
            save_folder_sixd += seq_id_sixd + '/'
            if not os.path.exists(save_folder_sixd):
                os.makedirs(save_folder_sixd)
            # pose_log_sixd = open(save_folder_sixd + filename_sixd, "w+")
            filename_sixd = img_id_sixd + '_' + obj_id_sixd + '.yml'
            """
            filename_sixd = "bop_results.csv"
            pose_log_sixd = open(str(Path(self.log_dir) / filename_sixd), "a")
            """
            pose_log_sixd.write('run_time: -1 \n')
            pose_log_sixd.write('ests: \n')
            str_score = '- {score: 1.00000000, '
            str_R = 'R: [{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}], ' \
                .format(self.rot_bar[0, 0], self.rot_bar[0, 1], self.rot_bar[0, 2],
                        self.rot_bar[1, 0], self.rot_bar[1, 1], self.rot_bar[1, 2],
                        self.rot_bar[2, 0], self.rot_bar[2, 1], self.rot_bar[2, 2])

            str_t = 't: [{:.8f}, {:.8f}, {:.8f}]'.format(self.trans_bar[0] * 1000.0,
                                                         self.trans_bar[1] * 1000.0,
                                                         self.trans_bar[2] * 1000.0)
            
            """
            R = '{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}' \
                .format(self.rot_bar[0, 0], self.rot_bar[0, 1], self.rot_bar[0, 2],
                        self.rot_bar[1, 0], self.rot_bar[1, 1], self.rot_bar[1, 2],
                        self.rot_bar[2, 0], self.rot_bar[2, 1], self.rot_bar[2, 2])

            t = '{:.8f} {:.8f} {:.8f}'.format(self.trans_bar[0] * 1000,
                                              self.trans_bar[1] * 1000,
                                              self.trans_bar[2] * 1000)

            # self.sixd_log += f"{seq_id_sixd}, {img_id_sixd}, {obj_id_sixd}, 1.000, {R}, {t}\n"
            # pose_log_sixd.write(str_score + str_R + str_t + '}')
            pose_log_sixd.write(f"{seq_id_sixd}, {img_id_sixd}, {obj_id_sixd}, 1.000, {R}, {t}, {self.time_elapse}\n")
            pose_log_sixd.close()


    def display_overall_result(self):
        print('trans mean error = ', np.mean(np.asarray(self.log_err_t)))
        print('trans RMSE (x) = ', np.sqrt(np.mean(np.asarray(self.log_err_tx) ** 2)) * 1000)
        print('trans RMSE (y) = ', np.sqrt(np.mean(np.asarray(self.log_err_ty) ** 2)) * 1000)
        print('trans RMSE (z) = ', np.sqrt(np.mean(np.asarray(self.log_err_tz) ** 2)) * 1000)
        print('rot RMSE (x) = ', np.sqrt(np.mean(np.asarray(self.log_err_rx) ** 2)) * 57.3)
        print('rot RMSE (y) = ', np.sqrt(np.mean(np.asarray(self.log_err_ry) ** 2)) * 57.3)
        print('rot RMSE (z) = ', np.sqrt(np.mean(np.asarray(self.log_err_rz) ** 2)) * 57.3)
        print('rot mean error = ', np.mean(np.asarray(self.log_err_r)))


    def eval_dataset(self, val_dataset, sequence, show_vis=True):
            self.log_err_r = []
            self.log_err_t = []
            self.log_dir = str(Path(self.log_dir) / 'eval' / str(time.time_ns()))
            if not Path(self.log_dir).exists():
                Path(self.log_dir).mkdir(parents=True, exist_ok=True)

            val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                        shuffle=False, num_workers=0)
            steps = len(val_dataset)
            step = 0

            vis_img_list = []
            for inputs in tqdm(val_generator, desc=f"Evaluation of {val_dataset.dataset_type}"):
                if val_dataset.dataset_type == 'tless':
                    images, depths, poses_gt, intrinsics, class_mask, \
                    file_name, _, bbox = inputs

                    self.data_intrinsics = intrinsics[0].numpy()
                    self.intrinsics = intrinsics[0].numpy()
                    self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
                    self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
                    self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
                    self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]
                    self.renderer.set_intrinsics(self.intrinsics, im_w=images.size(2), im_h=images.size(1))

                    self.data_with_est_center = True
                    self.data_with_gt = True

                    # ground truth for visualization
                    pose_gt = poses_gt.numpy()[0, :, :]
                    self.gt_t = pose_gt[:3, 3]
                    self.gt_rotm = pose_gt[:3, :3]
                    gt_center = np.matmul(intrinsics, self.gt_t)
                    if gt_center.shape[0] == 1:
                        gt_center = gt_center[0]
                    gt_center = gt_center / gt_center[2]
                    self.gt_uv = gt_center[:2]
                    self.gt_z = self.gt_t[2]


                    # Note: we are only predicitng orientation estimates
                    self.trans_bar = self.gt_t
                    self.uv_bar = self.gt_uv
                else:
                    print('*** INCORRECT DATASET SETTING! ***')
                    break

                self.step = step

                if self.modality == 'rgbd':
                    depth_data = depths[0]
                else:
                    depth_data = None

                torch.cuda.synchronize()
                time_start = time.time()

                self.estimate_orientation(images[0].detach(),
                                         self.data_intrinsics)
                
                torch.cuda.synchronize()
                self.time_elapse = time.time() - time_start
                print('[Orientation estimation] fps = ', 1 / self.time_elapse)

                # logging
                self.display_result(step, steps)
                self.save_log(sequence, file_name, tless=(self.obj_ctg == 'tless'))

                # visualization
                if show_vis:
                    image_disp = images[0].float().numpy()
                    image_est_render, _ = self.renderer.render_pose(self.intrinsics,
                                                                    self.trans_bar,
                                                                    self.rot_bar,
                                                                    self.target_obj_idx)

                    image_est_disp = image_est_render[0].permute(1, 2, 0).cpu().numpy()
                    obj_mask = np.all((image_est_disp[:, :] != np.zeros(3)), axis=-1)
                    image_est_disp[obj_mask] = seeded_colors

                    image_disp = 0.4 * image_disp + 0.6 * image_est_disp

                    vis_img_list.append(image_disp * 255)


                if step == steps-1:
                    break
                step += 1

            self.display_overall_result()
            self.log_video(vis_img_list)
