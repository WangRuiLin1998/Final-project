#
# 2021年内存处理捕捉图像版
#
from ctypes import *
import math
import random
import cv2
import os
import socket
import time
import numpy
import numpy as np

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image

ndarray_image = lib.ndarray_to_image 
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)] 
ndarray_image.restype = IMAGE


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect_im(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
# 1
#    im = load_image(image, 0, 0)
# 2
    im = nparray_to_image(image)
# please refer to : https://lwplw.blog.csdn.net/article/details/84566954
# please refer to : https://github.com/pjreddie/darknet/issues/289

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def detect_and_boxing_im(net, meta, b_path,im,color=(0.255, 255), line_type=1):
# def detect_and_boxing_im(net, meta, b_path,save_path,color=(0.255, 255), line_type=1):
#
#    im = cv2.imread(save_path)
#
# 1
#    r = detect_im(net, meta, b_path)
# 2
    r = detect_im(net, meta, im) 
    if not len(r) > 0:
        print("nothing detected in this picture!")
        cv2.imshow('Nothing',im)
    else:
        for i in range(len(r)):
            box_i = r[i]
            label_i = box_i[0]
            prob_i = box_i[1]
            x_ = box_i[2][0]
            y_ = box_i[2][1]
            w_ = box_i[2][2]
            h_ = box_i[2][3]
            text_ = str(label_i) + "," + str(round(prob_i, 3))
            cv2.rectangle(im, (int(x_ - w_ / 2), int(y_ - h_ / 2)),(int(x_ + w_ / 2), int(y_ + h_ / 2)),color, line_type)
            cv2.putText(im, text_, (int(x_ - w_ / 2 - 5), int(y_ - h_ / 2 - 5)), cv2.FONT_HERSHEY_DUPLEX, 0.7, color,2)
#            cv2.imwrite(save_path, image)
            cv2.imshow('Detecting',im)
            print("boxing ", i, " found ", label_i, "with prob = ", prob_i, ", finished!")
            print("box position is :", box_i[2])

def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
        
def Receive_and_detectVideo():
    net = load_net(b"yolov31.cfg", b"yolov3.weights", 0)
    meta = load_meta(b"cfg/coco.data")
#
    b_path = b"temp.jpg"
    save_path = "temp.jpg"
#     
    address = ('192.168.10.109',9999)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(address)
    s.listen(1)
    conn, addr = s.accept()
    print('connect from:'+str(addr))
    while 1:
        start = time.time()
        length = recvall(conn,16)
        stringData = recvall(conn, int(length))
        data = numpy.frombuffer(stringData, numpy.uint8)
        decimg=cv2.imdecode(data,cv2.IMREAD_COLOR)
# ---------------------------------------------------------------
        cv2.imshow('connect from:'+str(addr),decimg)
#
#        cv2.imwrite(save_path, decimg)
#
        detect_and_boxing_im(net, meta, b_path, decimg)    
# ---------------------------------------------------------------    
        end = time.time()
        seconds = end - start
        fps  = 1/seconds;
        conn.send(bytes(str(int(fps)),encoding='utf-8'))
        k = cv2.waitKey(10)&0xff
        if k == 27:
           break
    s.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Receive_and_detectVideo()
