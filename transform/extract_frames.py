import pylab
import imageio
import pickle


imageio.plugins.ffmpeg.download()


def show_frames(vid):
    nums = [10, 30]
    for num in nums:
        image = vid.get_data(num)

        print(['y', 'x', 'color'])
        print(image.shape)
        print([len(image), len(image[0]), len(image[0][0])])

        fig = pylab.figure()
        fig.suptitle('image #{}'.format(num), fontsize=20)
        pylab.imshow(image)

    pylab.show()


def show_timestamp(vid):
    print(vid._meta['fps'])
    print(vid.get_meta_data()['fps'])

    for num, image in enumerate(vid.iter_data()):
        if num % round(vid._meta['fps']):
            continue

        timestamp = float(num) / vid.get_meta_data()['fps']
        print(timestamp)

        fig = pylab.figure()
        fig.suptitle('image #{}, timestamp={}'.format(num, timestamp), fontsize=20)
        pylab.imshow(image)

        pylab.show()


def get_frames(vid):
    print(vid._meta['fps'])
    print(vid.get_meta_data()['fps'])

    frames = []

    for num, image in enumerate(vid.iter_data()):
        if num % round(vid._meta['fps']):
            continue
        if num > 300:
            break

        timestamp = float(num) / vid.get_meta_data()['fps']
        print(timestamp)

        frames.append(image)

    return frames


if __name__ == '__main__':
    filename = '../data/Surveillance Feed - Parking Lot.mp4'
    vid = imageio.get_reader(filename, 'ffmpeg')

    frames = get_frames(vid)
    with open('../data/test.pkl', 'w') as f:
        pickle.dump(frames, f)
