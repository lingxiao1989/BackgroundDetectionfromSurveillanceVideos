import cv2

data_root = "../data/"
vidcap = cv2.VideoCapture(data_root + 'Surveillance Feed - Parking Lot.mp4')

success, image = vidcap.read()
count = 0
max_count = 10

while count < max_count:
    success, image = vidcap.read()
    count += 1
    print 'Read a new frame: ', success
    if success:
        # save frame as JPEG file
        cv2.imwrite(data_root + "frame%d.jpg" % count, image)
