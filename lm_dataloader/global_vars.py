S3_CLIENT = None
MPU = None

def set_mpu(mpu):
    global MPU
    if MPU is None:
        MPU = mpu
    else:
        raise ValueError('mpu already initialized')

def get_mpu():
    global MPU
    if MPU is None:
        raise ValueError('mpu not initialized')
    return MPU