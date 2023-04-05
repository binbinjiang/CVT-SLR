import argparse

def get_parser():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='The pytorch implementation for Visual Alignment Constraint '
                                                    'for Continuous Sign Language Recognition.')

    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    parser.add_argument('--config', default='./configs/baseline.yaml', help='path to the configuration file')
    parser.add_argument('--random_fix', type=str2bool, default=True, help='fix random seed or not')
    parser.add_argument('--device', type=str, default=0, help='the indexes of GPUs for training or testing')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # processor
    parser.add_argument('--phase', default='train', help='can be train, test and features')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # debug
    parser.add_argument('--save-interval', type=int, default=200, help='the interval for storing models (#epochs)')
    parser.add_argument('--random-seed', type=int, default=0, help='the default value for random seed.')
    parser.add_argument('--eval-interval', type=int, default=100, help='the interval for evaluating models (#epochs)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--log-interval', type=int, default=20, help='the interval for printing messages (#iteration)')
    parser.add_argument('--evaluate-tool', default="python", help='sclite or python')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # feeder
    parser.add_argument('--feeder', default='dataloader_video.BaseFeeder', help='data loader will be used')
    parser.add_argument('--dataset', default=None, help='data loader will be used')
    parser.add_argument('--dataset-info', default=dict(), help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--feeder-args', default=dict(), help='the arguments of data loader')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args',type=dict, default=dict(),help='the arguments of model')
    parser.add_argument('--load-weights',default=None,help='load weights for network initialization')
    parser.add_argument('--restore-training-weights',default=None,help='load the whole weights for network continue training')
    parser.add_argument('--load-checkpoints',default=None,help='load checkpoints for continue training')
    parser.add_argument('--decode-mode',default="max",help='search mode for decode, max or beam')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # optim
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=8, help='test batch size')

    parser.add_argument("--use_seqAE", default="VAE", help='AE, VAE')

    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
