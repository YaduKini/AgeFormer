import argparse


def parse_settings():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        default='/home/guests/projects/ukbb/abdominal/nifti',
        type=str,
        help='path to ukbb dataset'
    )

    parser.add_argument(
        '--target_csv',
        default='/home/guests/projects/ukbb/yadu/age_imaging_filtered_train_val.csv',
        type=str,
        help='path to target csv'
    )

    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        type=float,
        help='initial learning rate'
    )

    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='number of jobs'
    )

    parser.add_argument(
        '--batch_size',
        default=2,
        type=int,
        help='batch size'
    )

    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='save model after'
    )

    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='total epochs'
    )

    parser.add_argument(
        '--input_x',
        default=224, # ukbb mri
        type=int,
        help='Input size of depth'
    )

    parser.add_argument(
        '--input_y',
        default=168, # ukbb mri
        type=int,
        help='Input size of height'
    )

    parser.add_argument(
        '--input_z',
        default=363, # ukbb mri
        type=int,
        help='Input size of width'
    )

    parser.add_argument(
        '--pretrain_path',
        default='pretrain/resnet_18.pth', # change this to ukbb. move pretrained models to my project folder
        type=str,
        help=
        'Path for pretrained model.'
    )

    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,
        help='Gpu id lists'
    )

    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | '
    )

    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)'
    )

    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        help='train phase or test phase'
    )

    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='If true, cuda is not used.'
    )

    parser.add_argument(
        "--num_classes",
        default=1,
        type=int,
        help="number of classes"
    )

    args = parser.parse_args()
    args.save_folder = "./home/guests/yadunandan_kini/yadu/medicalnet/sex_prediction/models/{}_{}".format(args.model, args.model_depth)

    return args
