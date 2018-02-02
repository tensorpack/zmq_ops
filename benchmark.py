#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: benchmark.py

import argparse
import time
import numpy as np
import sys
import tqdm

import zmq
import tensorflow as tf

from zmq_ops import dump_arrays, ZMQPullSocket

PIPE = 'ipc://@testpipe'
TQDM_BAR_FMT = '{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
TQDM_BAR_LEN = 1000

def send():
    """ We use float32 data to pressure the system. In practice you'd use uint8 images."""
    data = [
        np.random.rand(64, 224, 224, 3).astype('float32'),
        (np.random.rand(64)*100).astype('int32')
    ]   # 37MB per data
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.set_hwm(args.hwm)
    socket.connect(PIPE)

    try:
        while True:
            with tqdm.trange(TQDM_BAR_LEN, ascii=True, bar_format=TQDM_BAR_FMT) as pbar:
                for k in range(TQDM_BAR_LEN):
                    socket.send(dump_arrays(data), copy=False)
                    pbar.update(1)
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        if not ctx.closed:
            ctx.destroy(0)
        sys.exit()


def recv():
    sock = ZMQPullSocket(PIPE, [tf.float32, tf.int32], args.hwm)
    fetches = []
    for k in range(8):  # 8 GPUs pulling together in one sess.run call
        fetches.extend(sock.pull())
    fetch_op = tf.group(*fetches)

    with tf.Session() as sess:
        while True:
            with tqdm.trange(TQDM_BAR_LEN, ascii=True, bar_format=TQDM_BAR_FMT) as pbar:
                for k in range(TQDM_BAR_LEN // 8):
                    sess.run(fetch_op)
                    pbar.update(8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['send', 'recv'])
    parser.add_argument('--hwm', type=int, default=200)
    args = parser.parse_args()

    if args.task == 'send':
        send()
    elif args.task == 'recv':
        recv()
