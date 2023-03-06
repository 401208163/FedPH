# -*- coding: utf-8 -*-
# @Time    : 10/29/22 5:31 下午
# @Author  : Kuang Hangdong
# @File    : TimeConsumption.py
# @desc    :
import time

import numpy as np

from distro_paillier.distributed_paillier import NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PRIME_THRESHOLD, STATISTICAL_SECURITY_SECRET_SHARING, CORRECTNESS_PARAMETER_BIPRIMALITY
from distro_paillier.distributed_paillier import generate_shared_paillier_key
import diffprivlib as dp
from phe import paillier
import random

random.seed(42)
# paillier
public_key, private_key = paillier.generate_paillier_keypair()

# smpc
Key, pShares, qShares, N, PublicKey, LambdaShares, BetaShares, SecretKeyShares, theta = generate_shared_paillier_key(keyLength=128)

randomiser = dp.mechanisms.Gaussian(epsilon=1, delta=1, sensitivity=0.1)

message_384 = [random.random() for i in range(16)] * 24
message_33200 = [random.random() for i in range(16)] * 2075
timeList_384 = []
timeList_33200 = []

for i in range(10):
    start = time.time()
    randomised = [randomiser.randomise(number) for number in message_384]
    encryption_message_ = [PublicKey.encrypt(randomise) for randomise in randomised]
    message_smpc = [Key.decrypt(encryption_number, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta) for encryption_number in encryption_message_]
    stop = time.time()
    timeList_384.append(round(stop-start, 5))
timeList_384 = np.array(timeList_384)

for i in range(10):
    start = time.time()
    randomised = [randomiser.randomise(number) for number in message_33200]
    encryption_message_ = [PublicKey.encrypt(randomise) for randomise in randomised]
    message_smpc = [Key.decrypt(encryption_number, NUMBER_PLAYERS, CORRUPTION_THRESHOLD, PublicKey, SecretKeyShares, theta) for encryption_number in encryption_message_]
    stop = time.time()
    timeList_33200.append(round(stop-start, 5))
timeList_33200 = np.array(timeList_33200)

print(f"384 mean:{timeList_384.mean()},std{timeList_384.std()}")
print(f"33200 mean:{timeList_33200.mean()},std{timeList_33200.std()}")