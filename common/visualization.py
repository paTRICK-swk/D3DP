# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    # w = 1000
    # h = 1002

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


# def draw_3d_image(pred, gt, skeleton, azim, sub, act, cam):
#     frame = 200
#
#     pred = pred[:, :, frame]
#     gt = gt[frame]
#
#     pred = (pred - pred[:, :, 0:1]) * 1000
#     gt = (gt - gt[0:1]) * 1000
#     # data_3d[0:1]=0
#
#     # plt.axis("off")
#     # # plt.xlim(0,1000)
#     # # plt.ylim(0,1000)
#     # plt.imshow(image)
#     #
#
#     parents = skeleton.parents()
#
#     for t in range(pred.shape[0]):
#         fig = plt.figure()
#
#         ax = fig.add_subplot(111, projection='3d')
#         # ax.set_xlabel('x')
#         #
#         # ax.set_ylabel('y')
#         #
#         # ax.set_zlabel('z')
#
#         xy_radius = 1000
#         radius = 1500
#         ax.view_init(elev=15., azim=azim - 70)
#         ax.set_xlim3d([-xy_radius / 2, xy_radius / 2])
#         ax.set_zlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([-xy_radius / 2, xy_radius / 2])
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.dist = 8
#         ax.set_title("timestep %d" % t)  # , pad=35
#         # ax.set_title("Cross dataset")  # , pad=35
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         ax.get_zaxis().set_visible(False)
#         # ax.set_axis_off()
#
#         # ax.scatter(data_3d[:, 0], data_3d[:, 1],data_3d[:,2], s=1)
#
#
#         for j, j_parent in enumerate(parents):
#             if j_parent == -1:
#                 continue
#
#             #col = 'yellowgreen' if i in joints_right else 'midnightblue'
#
#             for h in range(pred.shape[1]):
#                 ax.plot([pred[t, h, j, 0], pred[t, h, j_parent, 0]],
#                         [pred[t, h, j, 1], pred[t, h, j_parent, 1]],
#                         [pred[t, h, j, 2], pred[t, h, j_parent, 2]], zdir='z', linestyle='--', linewidth=0.5)
#
#             col = 'blue'
#             ax.plot([gt[j, 0], gt[j_parent, 0]],
#                     [gt[j, 1], gt[j_parent, 1]],
#                     [gt[j, 2], gt[j_parent, 2]], zdir='z', c=col, linewidth=0.9)
#
#             # ax.annotate(s=str(i), x=data_2d[i,0], y=data_2d[i,1]-10,color='white', fontsize='3')
#
#
#         # plt.show()
#         plt.savefig("./plot/h36m/%s_%s_%d_frame%d_t%d.png" % (sub, act, cam, frame, t), bbox_inches="tight", pad_inches=0.0, dpi=300)
#         plt.close()

def draw_3d_image(pred_all, gt_all, skeleton, azim, sub, act, cam):
    #frame = 200
    for frame in range(gt_all.shape[0]):
        if frame %5 != 0:
            continue

        pred = pred_all[:, :, frame]
        gt = gt_all[frame]

        pred = (pred - pred[:, :, 0:1]) * 1000
        gt = (gt - gt[0:1]) * 1000
        # data_3d[0:1]=0

        # plt.axis("off")
        # # plt.xlim(0,1000)
        # # plt.ylim(0,1000)
        # plt.imshow(image)
        #

        parents = skeleton.parents()

        #for t in range(0,pred.shape[0]):
        for t in range(pred.shape[0]-1, pred.shape[0]):
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlabel('x')
            #
            # ax.set_ylabel('y')
            #
            # ax.set_zlabel('z')

            xy_radius = 500
            radius = 750
            azim_delta = 70
            ax.view_init(elev=15., azim=azim - azim_delta)
            #ax.view_init(elev=15., azim=azim)
            ax.set_xlim3d([-xy_radius, xy_radius])
            ax.set_zlim3d([-radius, radius])
            ax.set_ylim3d([-xy_radius, xy_radius])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 8
            #ax.set_title("timestep %d" % t)  # , pad=35
            # ax.set_title("Cross dataset")  # , pad=35
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_zaxis().set_visible(False)
            # ax.set_axis_off()

            # ax.scatter(data_3d[:, 0], data_3d[:, 1],data_3d[:,2], s=1)

            import matplotlib.colors as mcolors
            colors = list(mcolors.TABLEAU_COLORS.keys())

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                #col = 'yellowgreen' if i in joints_right else 'midnightblue'

                for h in range(pred.shape[1]):
                    ax.plot([pred[t, h, j, 0], pred[t, h, j_parent, 0]],
                            [pred[t, h, j, 1], pred[t, h, j_parent, 1]],
                            [pred[t, h, j, 2], pred[t, h, j_parent, 2]], zdir='z', linestyle='--', linewidth=0.5, c=mcolors.TABLEAU_COLORS[colors[h]])

                col = 'blue'
                ax.plot([gt[j, 0], gt[j_parent, 0]],
                        [gt[j, 1], gt[j_parent, 1]],
                        [gt[j, 2], gt[j_parent, 2]], zdir='z', c=col, linewidth=0.9)

                # ax.annotate(s=str(i), x=data_2d[i,0], y=data_2d[i,1]-10,color='white', fontsize='3')


            # plt.show()
            plt.savefig("./plot/h36m/%s_%s_%d_frame%d_t%d_azim%d.png" % (sub, act, cam, frame, t, azim_delta), bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close()

def draw_3d_image_select(pred_all, gt_all, skeleton, azim, sub, act, cam, gt_2d, pred_2d):

    pred_mean = np.mean(pred_all, axis=1, keepdims=True) #t, 1, f_all, n, 3
    t_sz, h_sz = pred_all.shape[0], pred_all.shape[1]

    import torch
    gt_2d = torch.from_numpy(gt_2d)
    gt_2d = gt_2d.unsqueeze(0).unsqueeze(0).repeat(t_sz, h_sz, 1, 1, 1)
    errors_2d = torch.norm(pred_2d - gt_2d, dim=len(gt_2d.shape) - 1)  # t, h, f_all, n
    select_ind = torch.min(errors_2d, dim=1, keepdims=True).indices  # t, 1, f_all, n
    select_ind = select_ind.unsqueeze(-1).repeat(1, 1, 1, 1, 3) # t, 1, f_all, n, 3
    pred_all_torch = torch.from_numpy(pred_all) # t, h, f_all, n, 3
    pred_select = torch.gather(pred_all_torch, 1, select_ind) # t, 1, f_all, n, 3

    #frame = 200
    #for frame in range(1210,1245):
    for frame in range(gt_all.shape[0]):
        if frame %5 != 0:
            continue

        pred = pred_all[:, :, frame]
        pred_m = pred_mean[:, :, frame]
        pred_s = pred_select[:, :, frame]
        gt = gt_all[frame]

        pred = (pred - pred[:, :, 0:1]) * 1000
        pred_m = (pred_m - pred_m[:, :, 0:1]) * 1000
        pred_s = (pred_s - pred_s[:, :, 0:1]) * 1000
        gt = (gt - gt[0:1]) * 1000
        # data_3d[0:1]=0

        # plt.axis("off")
        # # plt.xlim(0,1000)
        # # plt.ylim(0,1000)
        # plt.imshow(image)
        #

        parents = skeleton.parents()

        for t in range(0,pred.shape[0]):
        #for t in range(pred.shape[0]-1, pred.shape[0]):
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlabel('x')
            #
            # ax.set_ylabel('y')
            #
            # ax.set_zlabel('z')

            xy_radius = 500
            radius = 750
            azim_delta = 70
            ax.view_init(elev=15., azim=azim - azim_delta)
            #ax.view_init(elev=15., azim=azim)
            ax.set_xlim3d([-xy_radius, xy_radius])
            ax.set_zlim3d([-radius, radius])
            ax.set_ylim3d([-xy_radius, xy_radius])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 8
            #ax.set_title("timestep %d" % t)  # , pad=35
            # ax.set_title("Cross dataset")  # , pad=35
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_zaxis().set_visible(False)
            # ax.set_axis_off()

            # ax.scatter(data_3d[:, 0], data_3d[:, 1],data_3d[:,2], s=1)

            import matplotlib.colors as mcolors
            if h_sz>10:
                color_ind = list(mcolors.XKCD_COLORS.keys())
                my_color = mcolors.XKCD_COLORS
            else:
                color_ind = list(mcolors.TABLEAU_COLORS.keys())
                my_color = mcolors.TABLEAU_COLORS

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                #col = 'yellowgreen' if i in joints_right else 'midnightblue'

                for h in range(pred.shape[1]):
                    ax.plot([pred[t, h, j, 0], pred[t, h, j_parent, 0]],
                            [pred[t, h, j, 1], pred[t, h, j_parent, 1]],
                            [pred[t, h, j, 2], pred[t, h, j_parent, 2]], zdir='z', linestyle='--', linewidth=0.5, c=my_color[color_ind[h]])

                col = 'blue'
                ax.plot([gt[j, 0], gt[j_parent, 0]],
                        [gt[j, 1], gt[j_parent, 1]],
                        [gt[j, 2], gt[j_parent, 2]], zdir='z', c=col, linewidth=0.9)

                col = '#1b4332' #green
                ax.plot([pred_m[t, 0, j, 0], pred_m[t, 0, j_parent, 0]],
                        [pred_m[t, 0, j, 1], pred_m[t, 0, j_parent, 1]],
                        [pred_m[t, 0, j, 2], pred_m[t, 0, j_parent, 2]], zdir='z', c=col, linewidth=0.7)

                col = '#c1121f' #red
                ax.plot([pred_s[t, 0, j, 0], pred_s[t, 0, j_parent, 0]],
                        [pred_s[t, 0, j, 1], pred_s[t, 0, j_parent, 1]],
                        [pred_s[t, 0, j, 2], pred_s[t, 0, j_parent, 2]], zdir='z', c=col, linewidth=0.7)

                # ax.annotate(s=str(i), x=data_2d[i,0], y=data_2d[i,1]-10,color='white', fontsize='3')


            # plt.show()
            plt.savefig("./plot/h36m/%s_%s_%d_frame%d_t%d_azim%d.png" % (sub, act, cam, frame, t, azim_delta), bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close()

def draw_3d_image_azim(pred_all, gt_all, skeleton, azim, sub, act, cam, azim_off=0):
    #frame = 200
    for frame in range(gt_all.shape[0]):
        if frame %4 != 0:
            continue

        pred = pred_all[:, :, frame]
        gt = gt_all[frame]

        pred = (pred - pred[:, :, 0:1]) * 1000
        gt = (gt - gt[0:1]) * 1000
        # data_3d[0:1]=0

        # plt.axis("off")
        # # plt.xlim(0,1000)
        # # plt.ylim(0,1000)
        # plt.imshow(image)
        #

        parents = skeleton.parents()

        for t in range(pred.shape[0]-1,pred.shape[0]):
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlabel('x')
            #
            # ax.set_ylabel('y')
            #
            # ax.set_zlabel('z')

            xy_radius = 1000
            radius = 1500
            #ax.view_init(elev=15., azim=azim - 70)
            ax.view_init(elev=15., azim=azim+azim_off)
            ax.set_xlim3d([-xy_radius / 2, xy_radius / 2])
            ax.set_zlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([-xy_radius / 2, xy_radius / 2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 8
            ax.set_title("timestep %d" % t)  # , pad=35
            # ax.set_title("Cross dataset")  # , pad=35
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_zaxis().set_visible(False)
            # ax.set_axis_off()

            # ax.scatter(data_3d[:, 0], data_3d[:, 1],data_3d[:,2], s=1)


            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                #col = 'yellowgreen' if i in joints_right else 'midnightblue'

                for h in range(pred.shape[1]):
                    ax.plot([pred[t, h, j, 0], pred[t, h, j_parent, 0]],
                            [pred[t, h, j, 1], pred[t, h, j_parent, 1]],
                            [pred[t, h, j, 2], pred[t, h, j_parent, 2]], zdir='z', linestyle='--', linewidth=0.5)

                col = 'blue'
                ax.plot([gt[j, 0], gt[j_parent, 0]],
                        [gt[j, 1], gt[j_parent, 1]],
                        [gt[j, 2], gt[j_parent, 2]], zdir='z', c=col, linewidth=0.9)

                # ax.annotate(s=str(i), x=data_2d[i,0], y=data_2d[i,1]-10,color='white', fontsize='3')


            # plt.show()
            plt.savefig("./plot/h36m/%s_%s_%d_frame%d_t%d_azim%d.png" % (sub, act, cam, frame, t, azim_off), bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close()

def draw_3d_image_azim_ind(pred_all, gt_all, skeleton, azim, sub, act, cam, azim_off=0, select_ind=None, min_ind=None):
    #frame = 200
    for frame in range(gt_all.shape[0]):
        if frame %10 != 0:
            continue

        pred = pred_all[:, :, frame]
        gt = gt_all[frame]

        pred = (pred - pred[:, :, 0:1]) * 1000
        gt = (gt - gt[0:1]) * 1000
        # data_3d[0:1]=0

        # plt.axis("off")
        # # plt.xlim(0,1000)
        # # plt.ylim(0,1000)
        # plt.imshow(image)
        #

        parents = skeleton.parents()

        for t in range(0,pred.shape[0]):
            if t % 2 != 0:
                continue
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlabel('x')
            #
            # ax.set_ylabel('y')
            #
            # ax.set_zlabel('z')

            xy_radius = 1000
            radius = 1500
            #ax.view_init(elev=15., azim=azim - 70)
            ax.view_init(elev=15., azim=azim+azim_off)
            ax.set_xlim3d([-xy_radius / 2, xy_radius / 2])
            ax.set_zlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([-xy_radius / 2, xy_radius / 2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 8
            ax.set_title("timestep %d" % t)  # , pad=35
            # ax.set_title("Cross dataset")  # , pad=35
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_zaxis().set_visible(False)
            # ax.set_axis_off()

            for jj in range(17):
                min_ind_a = min_ind[t, 0, frame, jj]
                select_ind_a = select_ind[t, 0, frame, jj]
                ax.scatter(pred[t, select_ind_a, jj, 0], pred[t, select_ind_a, jj, 1], pred[t, select_ind_a, jj, 2],
                           s=0.5, c='g', zorder=10)
                ax.scatter(pred[t, min_ind_a, jj, 0], pred[t, min_ind_a, jj, 1], pred[t, min_ind_a, jj, 2], s=2, c='r',zorder=4)

                ax.text(s=str(min_ind_a.numpy()), x=pred[t, min_ind_a, jj, 0] + 10, y=pred[t, min_ind_a, jj, 1],z=pred[t, min_ind_a, jj, 2] + 20, color='r', fontsize='3')
                ax.text(s=str(select_ind_a.numpy()), x=pred[t, select_ind_a, jj, 0] - 10, y=pred[t, select_ind_a, jj, 1],z=pred[t, select_ind_a, jj, 2] + 20, color='g', fontsize='3')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                #col = 'yellowgreen' if i in joints_right else 'midnightblue'

                for h in range(pred.shape[1]):
                    ax.plot([pred[t, h, j, 0], pred[t, h, j_parent, 0]],
                            [pred[t, h, j, 1], pred[t, h, j_parent, 1]],
                            [pred[t, h, j, 2], pred[t, h, j_parent, 2]], zdir='z', linestyle='--', linewidth=0.5)

                col = 'blue'
                ax.plot([gt[j, 0], gt[j_parent, 0]],
                        [gt[j, 1], gt[j_parent, 1]],
                        [gt[j, 2], gt[j_parent, 2]], zdir='z', c=col, linewidth=0.9)

                # ax.annotate(s=str(i), x=data_2d[i,0], y=data_2d[i,1]-10,color='white', fontsize='3')


            # plt.show()
            plt.savefig("./plot/h36m/%s_%s_%d_frame%d_t%d_azim%d.png" % (sub, act, cam, frame, t, azim_off), bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close()

def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, newpose=None):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    if newpose is not None:
        fig = plt.figure(figsize=(size * (1 + len(poses) + len(newpose)), size))
        ax_in = fig.add_subplot(1, 1 + len(poses) + len(newpose), 1)
    else:
        fig = plt.figure(figsize=(size * (1 + len(poses)), size))
        ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    if newpose is not None:
        axnew = fig.add_subplot(1, 1 + len(poses) + len(newpose), 2, projection='3d')
        axnew.view_init(elev=15., azim=azim)
        axnew.set_xlim3d([-radius / 2, radius / 2])
        axnew.set_zlim3d([0, radius])
        axnew.set_ylim3d([-radius / 2, radius / 2])
        try:
            axnew.set_aspect('equal')
        except NotImplementedError:
            axnew.set_aspect('auto')
        axnew.set_xticklabels([])
        axnew.set_yticklabels([])
        axnew.set_zticklabels([])
        axnew.dist = 7.5
        axnew.set_title('PoseFormer') #, pad=35
        ax_3d.append(axnew)
        lines_3d.append([])
        trajectories.append(newpose[:, 0, [0, 1]])

    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:] # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        if newpose is not None:
            newpose = newpose[input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        if newpose is not None:
            newpose = downsample_tensor(newpose, downsample)
            for idx in range(len(poses)+len(newpose)):
                poses[idx] = downsample_tensor(poses[idx], downsample)
                trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        else:
            for idx in range(len(poses)):
                poses[idx] = downsample_tensor(poses[idx], downsample)
                trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'orange' if j in skeleton.joints_right() else 'green'
                
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
            # Plot 2D keypoints
            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                # Plot 2D keypoints
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j-1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
            # Plot 2D keypoints
            points.set_offsets(keypoints[i])

        print('{}/{}      '.format(i, limit), end='\r')


    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
