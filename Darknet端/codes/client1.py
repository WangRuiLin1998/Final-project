import socket
import cv2
import numpy
import time

def SendVideo():
    address = ('192.168.10.109', 9999)
    try:
        sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),60]

    while ret:
        time.sleep(0.01)
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()
        sock.send(str.encode(str(len(stringData)).ljust(16)));
        sock.send(stringData);
        receive = sock.recv(1024)
        if len(receive):print(str(receive,encoding='utf-8'))
        ret, frame = capture.read()
        if cv2.waitKey(10) == 27:
           break
    sock.close()

if __name__ == '__main__':
    SendVideo()

