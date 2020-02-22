
import cv2
hog = cv2.HOGDescriptor()
image = imutils.resize(image, width=min(400, image.shape[1]))

clone = image.copy()
(rects, weights) = HOGCV.detectMultiScale(image, winStride=(8, 8),padding=(32, 32), scale=1.05)

    # Applies non-max supression from imutils package to kick-off overlapped
    # boxes
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
result = non_max_suppression(rects, probs=None, overlapThresh=0.65)
