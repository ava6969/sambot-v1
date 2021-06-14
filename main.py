from utils import init
from trainer.arguments import get_args
from train import train


def test():
    pass


if __name__ == '__main__':

    args = get_args()
    startup_data = init(args)
    train(startup_data, args, 0.4)
    test()
