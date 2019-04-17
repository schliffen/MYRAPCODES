#
# Creating Fit Generator
#
from skimage.io import imread
from skimage.transform import resize
import numpy as np
#
class seq_generator(Sequence):
    
    def __init__(self, frame_filename, labels, batch_size):
        self.frame_filename, self.labels = frame_filename, labels
        self.batch_size = batch_size
    
    def __len__(self):
        return np.ceil(len(self.frame_filename)/float(self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.frame_filename[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
    
        return np.array([
            resize(imread(file_name), (200,200)) for file_name in batch_x]), np.array(batch_y)
    


seq_train_batch_generator = seq_generator(training_filenames, GT_training, batch_size)
seq_validation_batch_generator = seq_generator(validation_filename, GT_validation, batch_size)

model.fit_generator(generator=seq_train_batch_generator,
                              steps_per_epoch= (num_train_sample // batch_size),
                              epochs=num_epoches,
                              verbose=1,
                              validation_data= seq_validation_batch_generator,
                              validation_steps=(num_validation_sample // batch_size),
                              use_multiprocessing= True,
                              workers=8,
                              max_queue_size=16)
#
# manual approach -> to be develpoed later 
#
# this function will use for seq generation!
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)