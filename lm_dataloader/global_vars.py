S3_CLIENT = None
MPU = None


def set_mpu(mpu):
    global MPU
    if MPU is None:
        MPU = mpu
    else:
        raise ValueError("mpu already initialized")


def get_mpu():
    global MPU
    if MPU is None:
        print("mpu not initialized, please call set_mpu() first")
        print("Returning None")
    return MPU