# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from common.utils import read_video


def ckpt_time(ckpt=None, display=0, desc=''):
    if not ckpt:
        return time.time()
    else:
        if display:
            print(desc + ' consume time {:0.4f}'.format(time.time() - float(ckpt)))
        return time.time() - float(ckpt), time.time()


def set_equal_aspect(ax, data):
    """
    Create white cubic bounding box to make sure that 3d axis is in equal aspect.
    :param ax: 3D axis
    :param data: shape of(frames, 3), generated from BVH using convert_bvh2dataset.py
    """
    X, Y, Z = data[..., 0], data[..., 1], data[..., 2]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
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
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    # prevent wired error
    _ = Axes3D.__class__.__name__

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 12.5
        ax.set_title(title, y = 0.9)  # , pad=35
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
        for f in read_video(input_video_path, fps=None, skip=input_video_skip):
            all_frames.append(f)

        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
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
    pbar = tqdm(total=limit)

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                # if len(parents) == keypoints.shape[1] and 1 == 2:
                #     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                #     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                #                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'yellowgreen' if j in skeleton.joints_right() else 'midnightblue'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            points = ax_in.scatter(*keypoints[i].T, 5, color='red', edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                # if len(parents) == keypoints.shape[1] and 1 == 2:
                #     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                #                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]])) # Hotfix matplotlib's bug. https://github.com/matplotlib/matplotlib/pull/20555
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')

            points.set_offsets(keypoints[i])

        pbar.update()

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=limit, interval=1000.0 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=60, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    pbar.close()
    plt.close()

def draw_3d_image(pred_all, skeleton, azim, video_name=""):
    #frame = 200
    for frame in range(pred_all.shape[2]):
        if frame %2 != 0:
            continue

        pred = pred_all[:, :, frame]*1000
        #gt = gt_all[frame]

        #pred = (pred - pred[:, :, 0:1])
        #gt = (gt - gt[0:1]) * 1000
        # data_3d[0:1]=0

        # plt.axis("off")
        # # plt.xlim(0,1000)
        # # plt.ylim(0,1000)
        # plt.imshow(image)
        #

        parents = skeleton.parents()

        for t in range(pred.shape[0]-1, pred.shape[0]):
        #for t in range(0,pred.shape[0]):
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlabel('x')
            #
            # ax.set_ylabel('y')
            #
            # ax.set_zlabel('z')

            xy_radius = 1500
            radius = 1500
            ax.view_init(elev=15., azim=azim)
            #ax.view_init(elev=15., azim=azim)
            ax.set_zlim3d([0, radius])
            #ax.set_zlim3d([-radius / 2, radius / 2])
            ax.set_xlim3d([-xy_radius / 2, xy_radius / 2])
            ax.set_ylim3d([-xy_radius / 2, xy_radius / 2])
            # ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            # ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 8
            #ax.set_title("frame %d" % frame)  # , pad=35
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

                #col = 'blue'
                # ax.plot([gt[j, 0], gt[j_parent, 0]],
                #         [gt[j, 1], gt[j_parent, 1]],
                #         [gt[j, 2], gt[j_parent, 2]], zdir='z', c=col, linewidth=0.9)

                # ax.annotate(s=str(i), x=data_2d[i,0], y=data_2d[i,1]-10,color='white', fontsize='3')


            # plt.show()
            plt.savefig("./outputs/%s/frame%d_t%d.png" % (video_name, frame, t), bbox_inches="tight", pad_inches=0.0, dpi=300)
            plt.close()

def render_animation_test(keypoints, poses, skeleton, fps, bitrate, azim, output, viewport, limit=-1, downsample=1, size=6, input_video_frame=None,
                          input_video_skip=0, num=None):
    t0 = ckpt_time()
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    fig.add_subplot(121)
    plt.imshow(input_video_frame)
    # 3D
    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(elev=15., azim=azim)
    # set 长度范围
    radius = 1.7
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5

    # lxy add
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()

    pos = poses['Reconstruction'][-1]
    _, t1 = ckpt_time(t0, desc='1 ')
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        if len(parents) == keypoints.shape[1]:
            color_pink = 'pink'
            if j == 1 or j == 2:
                color_pink = 'black'

        col = 'red' if j in skeleton.joints_right() else 'black'
        # 画图3D
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    #  plt.savefig('test/3Dimage_{}.png'.format(1000+num))
    width, height = fig.get_size_inches() * fig.get_dpi()
    _, t2 = ckpt_time(t1, desc='2 ')
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    cv2.imshow('im', image)
    cv2.waitKey(5)
    _, t3 = ckpt_time(t2, desc='3 ')
    return image
