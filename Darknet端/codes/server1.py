import socket
import time
import cv2
import numpy

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def ReceiveVideo():
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
        cv2.imshow('Receiving from:'+str(addr),decimg)
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
    ReceiveVideo()
