import os
import time

from common.arguments_diffusion import parse_args
from common.camera import *
from common.generators_diffusion import *
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate_diffusion, add_path
#from common.inference_3d import *

from common.diffusionpose import D3DP

# from joints_detectors.openpose.main import generate_kpts as open_pose


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

add_path()


# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


def get_detector_2d(detector_name):
    def get_alpha_pose():
        from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
        return alpha_pose

    def get_hr_pose():
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
        return hr_pose

    detector_map = {
        'alpha_pose': get_alpha_pose,
        'hr_pose': get_hr_pose,
        # 'open_pose': open_pose
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 14, 15, 16]


def main(args):
    detector_2d = get_detector_2d(args.detector_2d)

    assert detector_2d, 'detector_2d should be in ({alpha, hr, open}_pose)'

    # 2D kpts loads or generate
    #args.input_npz = './outputs/alpha_pose_basketball_cut/basketball_cut.npz'

    #args.input_npz = './outputs/alpha_pose_courtyard_bodyScannerMotions_00_cut/courtyard_bodyScannerMotions_00_cut.npz'
    #args.input_npz = './outputs/alpha_pose_outdoors_freestyle_00/outdoors_freestyle_00.npz'
    #args.input_npz = './outputs/alpha_pose_penn_weightlifting/penn_weightlifting.npz'
    args.input_npz = False
    if not args.input_npz:
        video_name = args.viz_video
        keypoints = detector_2d(video_name)
    else:
        npz = np.load(args.input_npz)
        keypoints = npz['kpts']  # (N, 17, 2)

    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # normlization keypoints  Suppose using the camera parameter
    import cv2
    cap = cv2.VideoCapture(args.viz_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=frame_width, h=frame_height)

    # model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
    #                           dense=args.dense)


    model_pos = D3DP(args,joints_left, joints_right, is_train=False, num_proposals=args.num_proposals, sampling_timesteps=args.sampling_timesteps)

    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()

    # if torch.cuda.is_available():
    #     model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('-------------- load data spends {:.2f} seconds'.format(ckpt))

    # load trained model
    # chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    # print('Loading checkpoint', chk_filename)
    # checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    # model_pos.load_state_dict(checkpoint['model_pos'])

    #model_dict = model_pos.state_dict()

    # no_refine_path = "checkpoint/no_refine_48_5137.pth"
    # pre_dict = torch.load(no_refine_path)
    # for key, value in pre_dict.items():
    #     name = key[7:]
    #     model_dict[name] = pre_dict[key]
    # model['trans'].load_state_dict(model_dict)

    chk_file_path = "./checkpoint/in_the_wild_best_epoch.bin"
    # chk_filename = args.resume or args.evaluate
    print('Loading checkpoint', chk_file_path)
    checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)


    ckpt, time2 = ckpt_time(time1)
    print('-------------- load 3D model spends {:.2f} seconds'.format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = args.number_of_frames
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    print('Rendering...')
    input_keypoints = keypoints.copy()
    print(input_keypoints.shape)
    gen = UnchunkedGenerator_Seq(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    prediction = evaluate_diffusion(gen, model_pos, return_predictions=True, receptive_field=receptive_field,
                                    bs=args.batch_size) # b, t, h, 243, j, c
    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = prediction.shape
    total_frame = input_keypoints.shape[0]
    prediction2 = np.empty((t_sz, h_sz, total_frame, 17, 3)).astype(np.float32)
    ### reshape prediction as ground truth
    if total_frame / receptive_field > total_frame // receptive_field:
        batch_num = (total_frame // receptive_field) + 1
        for i in range(batch_num - 1):
            prediction2[:, :, i * receptive_field:(i + 1) * receptive_field, :, :] = prediction[i, :, :, :, :, :]
        left_frames = total_frame - (batch_num - 1) * receptive_field
        prediction2[:, :, -left_frames:, :, :] = prediction[-1, :, :, -left_frames:, :, :]
        # prediction = prediction2
    elif total_frame / receptive_field == total_frame // receptive_field:
        batch_num = (total_frame // receptive_field)
        for i in range(batch_num):
            prediction2[:, :, i * receptive_field:(i + 1) * receptive_field, :, :] = prediction[i, :, :, :, :, :]

    # gen = Evaluate_Generator(128, None, None, [input_keypoints], args.stride,
    #                          pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation, shuffle=False,
    #                          kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    #prediction = val(args, gen, model_pos)

    save_path_dir = f'outputs/{args.video_name}'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    # save 3D joint points
    np.save(f'outputs/{args.video_name}/test_3d_{args.video_name}_output.npy', prediction2, allow_pickle=True)

    prediction2_c = prediction2.copy()
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction2 = camera_to_world(prediction2, R=rot, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction2[:, :, :, :, 2] -= np.min(prediction2[:, :, :, :, 2])

    np.save(f'outputs/{args.video_name}/test_3d_output_{args.video_name}_postprocess.npy', prediction2, allow_pickle=True)

    anim_output = {'Ours': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

    if not args.viz_output:
        args.viz_output = 'outputs/alpha_result.mp4'

    from common.visualization import draw_3d_image
    # render_animation(input_keypoints, anim_output,
    #                  Skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
    #                  limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
    #                  input_video_path=args.viz_video, viewport=(1000, 1002),
    #                  input_video_skip=args.viz_skip)

    draw_3d_image(prediction2, Skeleton(), np.array(70., dtype=np.float32), args.video_name)

    ckpt, time4 = ckpt_time(time3)
    print('total spend {:2f} second'.format(ckpt))


def inference_video(video_path, detector_2d):
    """
    Do image -> 2d points -> 3d points to video.
    :param detector_2d: used 2d joints detector. Can be {alpha_pose, hr_pose}
    :param video_path: relative to outputs
    :return: None
    """
    args = parse_args()

    args.detector_2d = detector_2d
    dir_name = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    args.video_name = basename[:basename.rfind('.')]
    args.viz_video = video_path
    # args.viz_export = f'{dir_name}/{args.detector_2d}_{video_name}_data.npy'
    args.viz_output = f'{dir_name}/{args.detector_2d}_{args.video_name}.mp4'
    # args.viz_limit = 20
    #args.input_npz = 'outputs/alpha_pose_test/test.npz'

    args.evaluate = 'pretrained_h36m_detectron_coco.bin'

    with Timer(video_path):
        main(args)


if __name__ == '__main__':
    inference_video('outputs/dancing.mp4', 'alpha_pose')
