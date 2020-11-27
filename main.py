import os

from helper import load_scenes_from_folder

from models.vgg_trainer import VGGTrainer
# from models.cnn_trainer import CNNTrainer
# from models.fully_connected_trainer import FullyConnectedTrainer


def main():

    # Load Scenes and pick one.
    path_to_scenes = os.path.join('.', 'dataset', 'scenes', 'scenes')
    scenes = load_scenes_from_folder(path_to_scenes)
    scene = scenes[5]

    # Choose a model
    model = VGGTrainer()  # FullyConnectedTrainer()

    model.create_dataset()
    model.train()
    model.detect_ships(scene)

    return


if __name__ == '__main__':
    main()
