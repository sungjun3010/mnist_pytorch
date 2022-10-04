from mnist.trainer import MNISTTrainer


def main():
    trainer = MNISTTrainer()
    trainer.fit()
    trainer.test()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
