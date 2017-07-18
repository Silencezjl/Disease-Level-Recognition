from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py


def write_gap(function_name, MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("data", image_size, shuffle=False,
                                              batch_size=64)
    # test_generator = gen.flow_from_directory("data/test", image_size, shuffle=False,
    #                                          batch_size=32, class_mode=None)

    train = model.predict_generator(train_generator, train_generator.nb_sample)
    # test = model.predict_generator(test_generator, test_generator.nb_sample)
    #
    # with open('filename.txt', 'w') as f:
    #     for i, fname in enumerate(test_generator.filenames):
    #         index = str(fname[fname.rfind('/')+1:fname.rfind('.')])
    #         f.write(index+'\n')

    with h5py.File("gap_%s.h5" % function_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("label", data=train_generator.classes)
        # h.create_dataset("label_map", data=train_generator.class_indices)

write_gap('ResNet50', ResNet50, (224, 224))
write_gap('InceptionV3', InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap('Xception', Xception, (299, 299), xception.preprocess_input)
