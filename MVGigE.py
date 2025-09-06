from ctypes import *
from GigECamera_Types import * 
import numpy as np 
import ctypes

MVGigE = windll.LoadLibrary(__file__[:__file__.rfind("\\")] + '\\MVGigE')

def MVInfo2Img(info):
    """
    convert from image info to np array
    :param info: pointer info from callback param
    :return: raw image in np array format
    """
    stFrameInfo = cast(info, POINTER(MV_IMAGE_INFO)).contents
    w = stFrameInfo.nSizeX
    h = stFrameInfo.nSizeY
    if stFrameInfo.nPixelType & 0x100000:
        ushort_buf = ctypes.cast(stFrameInfo.pImageBuffer, ctypes.POINTER(ctypes.c_uint16))
        image = np.ctypeslib.as_array(ushort_buf, shape=(h, w))
    else:
        image = np.ctypeslib.as_array(stFrameInfo.pImageBuffer, shape=(h, w))
    return image,stFrameInfo.nBlockId

def MV_info_to_image(cam_handle, info):
    """
    convert from image info to np array
    :param info: pointer info from callback param
    :return: M x N x 3 for color image, M x N for gray image
    """
    stFrameInfo = cast(info, POINTER(MV_IMAGE_INFO)).contents
    w = stFrameInfo.nSizeX
    h = stFrameInfo.nSizeY
    if stFrameInfo.nPixelType & 0x100000:
        if stFrameInfo.nPixelType == PixelFormat_Mono16:
            ushort_buf = ctypes.cast(stFrameInfo.pImageBuffer, ctypes.POINTER(ctypes.c_uint16))
            image = np.ctypeslib.as_array(ushort_buf, shape=(h, w))
        elif stFrameInfo.nPixelType in [PixelFormat_BayerBG16, PixelFormat_BayerRG16, PixelFormat_BayerGB16,
                                   PixelFormat_BayerGR16]:
            w = stFrameInfo.nSizeX
            h = stFrameInfo.nSizeY
            r, data = MVBayerToBGR16(cam_handle, stFrameInfo.pImageBuffer, w * 2 * 3, w, h, stFrameInfo.nPixelType)
            image = np.ctypeslib.as_array(data).reshape(h, w, 3)           
    else:
        if stFrameInfo.nPixelType == PixelFormat_Mono8:
            image = np.ctypeslib.as_array(stFrameInfo.pImageBuffer, shape=(h, w))
        elif stFrameInfo.nPixelType in [PixelFormat_BayerBG8, PixelFormat_BayerRG8, PixelFormat_BayerGB8,
                                   PixelFormat_BayerGR8]:
            r, data = MVBayerToBGR(cam_handle, stFrameInfo.pImageBuffer, w * 3, w, h, stFrameInfo.nPixelType, True)
            image = np.ctypeslib.as_array(data).reshape(h, w, 3)
    return image,stFrameInfo.nBlockId

    
def MVGetImgBuf(hCam):
    r, w = MVGetWidth(hCam)
    if r != MVST_SUCCESS:
        return r, None
        
    r, h = MVGetHeight(hCam)
    if r != MVST_SUCCESS:
        return r, None
        
    r, pixelformat = MVGetPixelFormat(hCam)
    if r != MVST_SUCCESS:
        return r, None
        
    if pixelformat == PixelFormat_Mono8:
        img = np.zeros((h, w), dtype=np.uint8)
    elif pixelformat == PixelFormat_Mono16:
        img = np.zeros((h, w), dtype=np.uint16)
    elif pixelformat in [PixelFormat_BayerBG8, PixelFormat_BayerRG8, PixelFormat_BayerGB8, PixelFormat_BayerGR8]:
        img = np.zeros((h, w, 3), dtype=np.uint8)
    elif pixelformat in [PixelFormat_BayerBG16, PixelFormat_BayerRG16, PixelFormat_BayerGB16, PixelFormat_BayerGR16]:
        img = np.zeros((h, w, 3), dtype=np.uint16)
    else:
        #print('未知pixel format : {:#X}'.format(pixelformat))
        return MVST_ERROR, None
        
    return MVST_SUCCESS, img
    
#MVInitLib()
def MVInitLib():
    """
     初始化函数库。在调用函数所有函数之前调用。
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVInitLib.restype = c_int
    res = MVGigE.MVInitLib()
    return res

#MVTerminateLib()
def MVTerminateLib():
    """
     退出函数库。在程序退出前调用，以释放资源。
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVTerminateLib.restype = c_int
    res = MVGigE.MVTerminateLib()
    return res

#MVUpdateCameraList()
def MVUpdateCameraList():
    """
        查找连接到计算机上的相机
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVUpdateCameraList.restype = c_int
    res = MVGigE.MVUpdateCameraList()
    return res

#MVGetNumOfCameras(int* pNumCams)
def MVGetNumOfCameras():
    """
        获取连接到计算机上的相机的数量
    :param pNumCams: 相机数量
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetNumOfCameras.argtype = (c_void_p)
    MVGigE.MVGetNumOfCameras.restype = c_int
    pNumCams = c_int()
    res = MVGigE.MVGetNumOfCameras(byref(pNumCams))
    return res, pNumCams.value

#MVGetCameraInfo(unsigned char idx, MVCamInfo* pCamInfo)
def MVGetCameraInfo(idx):
    """
        得到第idx个相机的信息。
    :param idx: idx从0开始，按照相机的IP地址排序，地址小的排在前面。
    :param pCamInfo: 相机的信息 (IP,MAC,SN,型号...)
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetCameraInfo.argtype = (c_ubyte, c_void_p)
    MVGigE.MVGetCameraInfo.restype = c_int
    pCamInfo = MVCamInfo()
    res = MVGigE.MVGetCameraInfo(c_ubyte(idx), byref(pCamInfo))
    return res, pCamInfo

#MVOpenCamByIndex(unsigned char idx, HANDLE* hCam)
def MVOpenCamByIndex(idx):
    """
        打开第idx个相机
    :param idx: idx从0开始，按照相机的IP地址排序，地址小的排在前面。
    :param hCam: 如果成功,返回的相机句柄
    :return:  MVST_INVALID_PARAMETER : idx取值不对
     *          MVST_ACCESS_DENIED      : 相机无法访问，可能正被别的软件控制
     *          MVST_ERROR              : 其他错误
     *          MVST_SUCCESS            : 成功
    """
    MVGigE.MVOpenCamByIndex.argtype = (c_ubyte, c_void_p)
    MVGigE.MVOpenCamByIndex.restype = c_int
    hCam = c_uint64()
    res = MVGigE.MVOpenCamByIndex(c_ubyte(idx), byref(hCam))
    return res, hCam.value

#MVOpenCamByUserDefinedName(char* name, HANDLE* hCam)
def MVOpenCamByUserDefinedName(name):
    """
        打开指定UserDefinedName的相机
    :param name: UserDefinedName。
    :param hCam: 如果成功,返回的相机句柄。如果失败，为NULL。
    :return:  
     *          MVST_ACCESS_DENIED      : 相机无法访问，可能正被别的软件控制
     *          MVST_ERROR              : 其他错误
     *          MVST_SUCCESS            : 成功
    """
    MVGigE.MVOpenCamByUserDefinedName.argtype = (c_char_p, c_void_p)
    MVGigE.MVOpenCamByUserDefinedName.restype = c_int
    cname = (c_char*16)() 
    hCam = c_uint64()
    print('len: ', len(name))
    for i in range(len(name)):
        cname[i] = c_char(name[i])
        
    res = MVGigE.MVOpenCamByUserDefinedName(cname, byref(hCam))
    return res, hCam.value

#MVOpenCamByIP( char *ip,HANDLE *hCam )
def MVOpenCamByIP(ip):
    """
        打开指定IP的相机
    :param ip: 相机的IP地址。
    :param hCam: 如果成功,返回的相机句柄。如果失败，为NULL。
    :return:  
     *          MVST_ACCESS_DENIED      : 相机无法访问，可能正被别的软件控制
     *          MVST_ERROR              : 其他错误
     *          MVST_SUCCESS            : 成功
    """
    MVGigE.MVOpenCamByIP.argtype = (c_char_p, c_void_p)
    MVGigE.MVOpenCamByIP.restype = c_int
    #ip = c_char()
    hCam = c_uint64()
    res = MVGigE.MVOpenCamByIP(ip.encode('ascii'), byref(hCam))
    return res, hCam.value

#MVCloseCam(HANDLE hCam)
def MVCloseCam(hCam):
    """
        关闭相机。断开和相机的连接。
    :param hCam: 相机的句柄
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVCloseCam.argtype = (c_uint64)
    MVGigE.MVCloseCam.restype = c_int
    res = MVGigE.MVCloseCam(c_uint64(hCam))
    return res

#MVGetWidth(HANDLE hCam, int* pWidth)
def MVGetWidth(hCam):
    """
        读取图像宽度
    :param hCam: 相机句柄
    :param pWidth: 图像宽度[像素]
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetWidth.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetWidth.restype = c_int
    pWidth = c_int()
    res = MVGigE.MVGetWidth(c_uint64(hCam), byref(pWidth))
    return res, pWidth.value

#MVGetWidthRange(HANDLE hCam, int* pWidthMin, int* pWidthMax)
def MVGetWidthRange(hCam):
    """
     读取图像宽度可设置的范围
    :param hCam: 相机句柄
    :param pWidthMin: 图像宽度可设置的最小值
    :param pWidthMax: 图像宽度可设置的最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetWidthRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetWidthRange.restype = c_int
    pWidthMin = c_int()
    pWidthMax = c_int()
    res = MVGigE.MVGetWidthRange(c_uint64(hCam), byref(pWidthMin), byref(pWidthMax))
    return res, pWidthMin.value, pWidthMax.value

#MVGetWidthInc(HANDLE hCam, int* pWidthInc)
def MVGetWidthInc(hCam):
    """
     读取图像宽度调整的步长
    :param hCam: 相机句柄
    :param pWidthInc: 图像宽度的调整的步长，即图像的宽度 = 最小宽度 + 步长 x 整数
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetWidthInc.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetWidthInc.restype = c_int
    pWidthInc = c_int()
    res = MVGigE.MVGetWidthInc(c_uint64(hCam), byref(pWidthInc))
    return res, pWidthInc.value

#MVSetWidth(HANDLE hCam, int nWidth)
def MVSetWidth(hCam, nWidth):
    """
     设置图像的宽度
    :param hCam: 相机句柄
    :param nWidth: 图像宽度，应该在宽度可设置范围之内，并且 = 最小宽度 + 步长 x 整数
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetWidth.argtype = (c_uint64, c_int)
    MVGigE.MVSetWidth.restype = c_int
    res = MVGigE.MVSetWidth(c_uint64(hCam), c_int(nWidth))
    return res

#MVGetHeight(HANDLE hCam, int* pHeight)
def MVGetHeight(hCam):
    """
        读取图像高度
    :param hCam: 相机句柄
    :param pHeight: 图像高度[像素]
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetHeight.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetHeight.restype = c_int
    pHeight = c_int()
    res = MVGigE.MVGetHeight(c_uint64(hCam), byref(pHeight))
    return res, pHeight.value

#MVGetHeightRange(HANDLE hCam, int* pHeightMin, int* pHeightMax)
def MVGetHeightRange(hCam):
    """
     读取图像高度可设置的范围
    :param hCam: 相机句柄
    :param pHeightMin: 图像高度可设置的最小值
    :param pHeightMax: 图像高度可设置的最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetHeightRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetHeightRange.restype = c_int
    pHeightMin = c_int()
    pHeightMax = c_int()
    res = MVGigE.MVGetHeightRange(c_uint64(hCam), byref(pHeightMin), byref(pHeightMax))
    return res, pHeightMin.value, pHeightMax.value

#MVSetHeight(HANDLE hCam, int nHeight)
def MVSetHeight(hCam, nHeight):
    """
     设置图像的高度
    :param hCam: 相机句柄
    :param nHeight: 图像高度，应该在高度可设置范围之内
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetHeight.argtype = (c_uint64, c_int)
    MVGigE.MVSetHeight.restype = c_int
    res = MVGigE.MVSetHeight(c_uint64(hCam), c_int(nHeight))
    return res

#MVGetOffsetX(HANDLE hCam, int* pOffsetX)
def MVGetOffsetX(hCam):
    """
     读取水平方向偏移量。图像宽度设置到小于最大宽度时，可以调整水平偏移量，设置采集窗口的水平起始位置。
    :param hCam: 
    :param pOffsetX: 水平偏移量
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetOffsetX.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetOffsetX.restype = c_int
    pOffsetX = c_int()
    res = MVGigE.MVGetOffsetX(c_uint64(hCam), byref(pOffsetX))
    return res, pOffsetX.value

#MVGetOffsetXRange(HANDLE hCam, int* pOffsetXMin, int* pOffsetXMax)
def MVGetOffsetXRange(hCam):
    """
     读取水平方向偏移量取值范围。
    :param hCam: 
    :param pOffsetXMin: 水平偏移量最小值
    :param pOffsetXMax: 水平偏移量最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetOffsetXRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetOffsetXRange.restype = c_int
    pOffsetXMin = c_int()
    pOffsetXMax = c_int()
    res = MVGigE.MVGetOffsetXRange(c_uint64(hCam), byref(pOffsetXMin), byref(pOffsetXMax))
    return res, pOffsetXMin.value, pOffsetXMax.value

#MVSetOffsetX(HANDLE hCam, int nOffsetX)
def MVSetOffsetX(hCam, nOffsetX):
    """
     设置水平方向偏移量。图像宽度设置到小于最大宽度时，可以调整水平偏移量，设置采集窗口的水平起始位置。
    :param hCam: 
    :param nOffsetX: 水平偏移量。应该在水平偏移量允许的范围之内。
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetOffsetX.argtype = (c_uint64, c_int)
    MVGigE.MVSetOffsetX.restype = c_int
    res = MVGigE.MVSetOffsetX(c_uint64(hCam), c_int(nOffsetX))
    return res

#MVGetOffsetY(HANDLE hCam, int* pOffsetY)
def MVGetOffsetY(hCam):
    """
     读取垂直方向偏移量。图像宽度设置到小于最大宽度时，可以调整垂直偏移量，设置采集窗口的垂直起始位置。
    :param hCam: 
    :param pOffsetY: 垂直偏移量
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetOffsetY.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetOffsetY.restype = c_int
    pOffsetY = c_int()
    res = MVGigE.MVGetOffsetY(c_uint64(hCam), byref(pOffsetY))
    return res, pOffsetY.value

#MVGetOffsetYRange(HANDLE hCam, int* pOffsetYMin, int* pOffsetYMax)
def MVGetOffsetYRange(hCam):
    """
     读取垂直方向偏移量取值范围。
    :param hCam: 
    :param pOffsetYMin: 垂直偏移量最小值
    :param pOffsetYMax: 垂直偏移量最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetOffsetYRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetOffsetYRange.restype = c_int
    pOffsetYMin = c_int()
    pOffsetYMax = c_int()
    res = MVGigE.MVGetOffsetYRange(c_uint64(hCam), byref(pOffsetYMin), byref(pOffsetYMax))
    return res, pOffsetYMin.value, pOffsetYMax.value

#MVSetOffsetY(HANDLE hCam, int nOffsetY)
def MVSetOffsetY(hCam, nOffsetY):
    """
     设置垂直方向偏移量。图像宽度设置到小于最大宽度时，可以调整垂直偏移量，设置采集窗口的垂直起始位置。
    :param hCam: 
    :param nOffsetY: 垂直偏移量。应该在垂直偏移量允许的范围之内。
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetOffsetY.argtype = (c_uint64, c_int)
    MVGigE.MVSetOffsetY.restype = c_int
    res = MVGigE.MVSetOffsetY(c_uint64(hCam), c_int(nOffsetY))
    return res

#MVGetPixelFormat(HANDLE hCam, MV_PixelFormatEnums* pPixelFormat)
def MVGetPixelFormat(hCam):
    """
        读取图像的像素格式
    :param hCam: 相机句柄
    :param pPixelFormat: 
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetPixelFormat.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetPixelFormat.restype = c_int
    pPixelFormat = c_uint()
    res = MVGigE.MVGetPixelFormat(c_uint64(hCam), byref(pPixelFormat))
    return res, pPixelFormat.value

def MVSetPixelFormat(hCam, pf):
    """
        设置图像的像素格式
    :param hCam: 相机句柄
    :param pf: PixelFormat
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetPixelFormat.argtype = (c_uint64, c_uint)
    MVGigE.MVSetPixelFormat.restype = c_int
    res = MVGigE.MVSetPixelFormat(c_uint64(hCam), pf)
    return res

#MVGetSensorTaps(HANDLE hCam, SensorTapsEnums* pSensorTaps)
def MVGetSensorTaps(hCam):
    """
        读取传感器的通道数
    :param hCam: 相机句柄
    :param pSensorTaps: 
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetSensorTaps.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetSensorTaps.restype = c_int
    pSensorTaps = c_uint()
    res = MVGigE.MVGetSensorTaps(c_uint64(hCam), byref(pSensorTaps))
    return res, pSensorTaps.value

#MVGetGain(HANDLE hCam, double* pGain)
def MVGetGain(hCam):
    """
        读取当前增益值
    :param hCam: 相机句柄
    :param pGain: 
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetGain.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGain.restype = c_int
    pGain = c_double()
    res = MVGigE.MVGetGain(c_uint64(hCam), byref(pGain))
    return res, pGain.value

#MVGetGainRange(HANDLE hCam, double* pGainMin, double* pGainMax)
def MVGetGainRange(hCam):
    """
        读取增益可以设置的范围
    :param hCam: 相机句柄
    :param pGainMin: 最小值
    :param pGainMax: 最大值
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetGainRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetGainRange.restype = c_int
    pGainMin = c_double()
    pGainMax = c_double()
    res = MVGigE.MVGetGainRange(c_uint64(hCam), byref(pGainMin), byref(pGainMax))
    return res, pGainMin.value, pGainMax.value

#MVSetGain(HANDLE hCam, double fGain)
def MVSetGain(hCam, fGain):
    """
        设置增益
    :param hCam: 相机句柄
    :param fGain: 增益
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetGain.argtype = (c_uint64, c_double)
    MVGigE.MVSetGain.restype = c_int
    res = MVGigE.MVSetGain(c_uint64(hCam), c_double(fGain))
    return res

#MVSetGainTaps(HANDLE hCam, double fGain, int nTap)
def MVSetGainTaps(hCam, fGain, nTap):
    """
        当相机传感器为多通道时，设置某个通道的增益
    :param hCam: 相机句柄
    :param fGain: 增益
    :param nTap: 通道。双通道[0,1],四通道[0,1,2,3]
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetGainTaps.argtype = (c_uint64, c_double, c_int)
    MVGigE.MVSetGainTaps.restype = c_int
    res = MVGigE.MVSetGainTaps(c_uint64(hCam), c_double(fGain), c_int(nTap))
    return res

#MVGetGainTaps(HANDLE hCam, double* pGain, int nTap)
def MVGetGainTaps(hCam, nTap):
    """
        当相机传感器为多通道时，读取某个通道的增益
    :param hCam: 相机句柄
    :param pGain: 
    :param nTap: 通道。双通道[0,1],四通道[0,1,2,3]
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetGainTaps.argtype = (c_uint64, c_void_p, c_int)
    MVGigE.MVGetGainTaps.restype = c_int
    pGain = c_double()
    res = MVGigE.MVGetGainTaps(c_uint64(hCam), byref(pGain), c_int(nTap))
    return res, pGain.value

#MVGetGainRangeTaps(HANDLE hCam, double* pGainMin, double* pGainMax, int nTap)
def MVGetGainRangeTaps(hCam, nTap):
    """
        当相机传感器为多通道时，读取某个通道的增益可设置的范围
    :param hCam: 相机句柄
    :param pGainMin: 增益最小值
    :param pGainMax: 增益最大值
    :param nTap: 通道。双通道[0,1],四通道[0,1,2,3]
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetGainRangeTaps.argtype = (c_uint64, c_void_p, c_void_p, c_int)
    MVGigE.MVGetGainRangeTaps.restype = c_int
    pGainMin = c_double()
    pGainMax = c_double()
    res = MVGigE.MVGetGainRangeTaps(c_uint64(hCam), byref(pGainMin), byref(pGainMax), c_int(nTap))
    return res, pGainMin.value, pGainMax.value

#MVGetWhiteBalance(HANDLE hCam, double* pRed, double* pGreen, double* pBlue)
def MVGetWhiteBalance(hCam):
    """
        读取当前白平衡系数
    :param hCam: 相机句柄
    :param pRed: 红色平衡系数
    :param pGreen: 绿色平衡系数
    :param pBlue: 蓝色平衡系数
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetWhiteBalance.argtype = (c_uint64, c_void_p, c_void_p, c_void_p)
    MVGigE.MVGetWhiteBalance.restype = c_int
    pRed = c_double()
    pGreen = c_double()
    pBlue = c_double()
    res = MVGigE.MVGetWhiteBalance(c_uint64(hCam), byref(pRed), byref(pGreen), byref(pBlue))
    return res, pRed.value, pGreen.value, pBlue.value

#MVGetWhiteBalanceRange(HANDLE hCam, double* pMin, double* pMax)
def MVGetWhiteBalanceRange(hCam):
    """
        读取白平衡设置的范围
    :param hCam: 相机句柄
    :param pMin: 系数最小值
    :param pMax: 系数最大值
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetWhiteBalanceRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetWhiteBalanceRange.restype = c_int
    pMin = c_double()
    pMax = c_double()
    res = MVGigE.MVGetWhiteBalanceRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value

#MVSetWhiteBalance(HANDLE hCam, double fRed, double fGreen, double fBlue)
def MVSetWhiteBalance(hCam, fRed, fGreen, fBlue):
    """
        设置白平衡系数
    :param hCam: 相机句柄
    :param fRed: 红色平衡系数
    :param fGreen: 绿色平衡系数
    :param fBlue: 蓝色平衡系数
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetWhiteBalance.argtype = (c_uint64, c_double, c_double, c_double)
    MVGigE.MVSetWhiteBalance.restype = c_int
    res = MVGigE.MVSetWhiteBalance(c_uint64(hCam), c_double(fRed), c_double(fGreen), c_double(fBlue))
    return res

#MVGetGainBalance(HANDLE hCam, int* pBalance)
def MVGetGainBalance(hCam):
    """
        读取是否通道自动平衡
    :param hCam: 相机句柄
    :param pBalance: 是否自动平衡
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetGainBalance.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGainBalance.restype = c_int
    pBalance = c_int()
    res = MVGigE.MVGetGainBalance(c_uint64(hCam), byref(pBalance))
    return res, pBalance.value

#MVSetGainBalance(HANDLE hCam, int nBalance)
def MVSetGainBalance(hCam, nBalance):
    """
        设置是否自动通道平衡
    :param hCam: 相机句柄
    :param nBalance: 是否自动通道平衡
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetGainBalance.argtype = (c_uint64, c_int)
    MVGigE.MVSetGainBalance.restype = c_int
    res = MVGigE.MVSetGainBalance(c_uint64(hCam), c_int(nBalance))
    return res

#MVGetExposureTime(HANDLE hCam, double* pExposuretime)
def MVGetExposureTime(hCam):
    """
        读取当前曝光时间
    :param hCam: 
    :param pExposuretime: 单位us
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetExposureTime.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetExposureTime.restype = c_int
    pExposuretime = c_double()
    res = MVGigE.MVGetExposureTime(c_uint64(hCam), byref(pExposuretime))
    return res, pExposuretime.value

#MVGetExposureTimeRange(HANDLE hCam, double* pExpMin, double* pExpMax)
def MVGetExposureTimeRange(hCam):
    """
      读取曝光时间的设置范围  
    :param hCam: 相机句柄
    :param pExpMin: 最短曝光时间 单位为us
    :param pExpMax: 最长曝光时间 单位为us
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetExposureTimeRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetExposureTimeRange.restype = c_int
    pExpMin = c_double()
    pExpMax = c_double()
    res = MVGigE.MVGetExposureTimeRange(c_uint64(hCam), byref(pExpMin), byref(pExpMax))
    return res, pExpMin.value, pExpMax.value

#MVSetExposureTime(HANDLE hCam,double nExp_us)
def MVSetExposureTime(hCam, nExp_us):
    """
        设置曝光时间
    :param hCam: 相机句柄
    :param nExp_us: 曝光时间 单位为us
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetExposureTime.argtype = (c_uint64, c_double)
    MVGigE.MVSetExposureTime.restype = c_int
    res = MVGigE.MVSetExposureTime(c_uint64(hCam), c_double(nExp_us))
    return res

#MVGetFrameRateRange(HANDLE hCam, double* pFpsMin, double* pFpsMax)
def MVGetFrameRateRange(hCam):
    """
        读取帧率可设置的范围
    :param hCam: 相机句柄
    :param pFpsMin: 最低帧率
    :param pFpsMax: 最高帧率
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetFrameRateRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetFrameRateRange.restype = c_int
    pFpsMin = c_double()
    pFpsMax = c_double()
    res = MVGigE.MVGetFrameRateRange(c_uint64(hCam), byref(pFpsMin), byref(pFpsMax))
    return res, pFpsMin.value, pFpsMax.value

#MVGetFrameRate(HANDLE hCam, double* fFPS)
def MVGetFrameRate(hCam):
    """
        读取当前帧率   
    :param hCam: 相机句柄
    :param fFPS: 帧率 帧/秒
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetFrameRate.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetFrameRate.restype = c_int
    fFPS = c_double()
    res = MVGigE.MVGetFrameRate(c_uint64(hCam), byref(fFPS))
    return res, fFPS.value

#MVSetFrameRate(HANDLE hCam, double fps)
def MVSetFrameRate(hCam, fps):
    """
        设置帧率
    :param hCam: 相机句柄
    :param fps: 帧率 帧/秒
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetFrameRate.argtype = (c_uint64, c_double)
    MVGigE.MVSetFrameRate.restype = c_int
    res = MVGigE.MVSetFrameRate(c_uint64(hCam), c_double(fps))
    return res

#MVStartGrab(HANDLE hCam, MVStreamCB StreamCB, long nUserVal)
def MVStartGrab(hCam, StreamCB, nUserVal):
    """
     开始采集图像
    :param hCam: 相机句柄
    :param StreamCB: 回调函数指针
    :param nUserVal: 用户数据，传递到回调函数的形参
    :return:    MVST_SUCCESS            : 成功
    """

    MVGigE.MVStartGrab.argtype = (c_uint64, MVStreamCB, c_ulonglong)
    MVGigE.MVStartGrab.restype = c_int
    res = MVGigE.MVStartGrab(c_uint64(hCam), StreamCB, c_ulonglong(nUserVal))
    return res

#MVStopGrab(HANDLE hCam)
def MVStopGrab(hCam):
    """
     停止采集图像
    :param hCam: 相机句柄
    :return:   MVST_SUCCESS         : 成功
    """
    MVGigE.MVStopGrab.argtype = (c_uint64)
    MVGigE.MVStopGrab.restype = c_int
    res = MVGigE.MVStopGrab(c_uint64(hCam))
    return res

#MVGetTriggerMode(HANDLE hCam, TriggerModeEnums* pMode)
def MVGetTriggerMode(hCam):
    """
        读取触发模式
    :param hCam: 相机句柄
    :param pMode: 触发模式 TriggerMode_Off,TriggerMode_On
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTriggerMode.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerMode.restype = c_int
    pMode = c_uint()
    res = MVGigE.MVGetTriggerMode(c_uint64(hCam), byref(pMode))
    return res, pMode.value

#MVSetTriggerMode(HANDLE hCam, TriggerModeEnums mode)
def MVSetTriggerMode(hCam, mode):
    """
        设置触发模式
    :param hCam: 相机句柄
    :param mode: 触发模式
        TriggerMode_Off：相机工作在连续采集模式，
        TriggerMode_On:相机工作在触发模式，需要有外触发信号或软触发指令才拍摄
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetTriggerMode.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerMode.restype = c_int
    res = MVGigE.MVSetTriggerMode(c_uint64(hCam), mode)
    return res

#MVGetTriggerSource(HANDLE hCam, TriggerSourceEnums* pSource)
def MVGetTriggerSource(hCam):
    """
        读取触发源
    :param hCam: 相机句柄
    :param pSource: 触发源，软触发或外触发
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTriggerSource.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerSource.restype = c_int
    pSource = c_uint()
    res = MVGigE.MVGetTriggerSource(c_uint64(hCam), byref(pSource))
    return res, pSource.value

#MVSetTriggerSource(HANDLE hCam, TriggerSourceEnums source)
def MVSetTriggerSource(hCam, source):
    """
        设置触发源
    :param hCam: 相机句柄
    :param source: 触发源
                    TriggerSource_Software：通过\c MVTriggerSoftware()函数触发。
                    TriggerSource_Line1：通过连接的触发线触发。
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetTriggerSource.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerSource.restype = c_int
    res = MVGigE.MVSetTriggerSource(c_uint64(hCam), source)
    return res

#MVGetTriggerActivation(HANDLE hCam, TriggerActivationEnums* pAct)
def MVGetTriggerActivation(hCam):
    """
        读取触发极性
    :param hCam: 相机句柄
    :param pAct: 
                    TriggerActivation_RisingEdge: 上升沿触发
                    TriggerActivation_FallingEdge: 下降沿触发
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTriggerActivation.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerActivation.restype = c_int
    pAct = c_uint()
    res = MVGigE.MVGetTriggerActivation(c_uint64(hCam), byref(pAct))
    return res, pAct.value

#MVSetTriggerActivation(HANDLE hCam, TriggerActivationEnums act)
def MVSetTriggerActivation(hCam, act):
    """
        当使用触发线触发时,设置是上升沿触发还是下降沿触发
    :param hCam: 
    :param act: 上升沿或下降沿
                    TriggerActivation_RisingEdge: 上升沿触发
                    TriggerActivation_FallingEdge: 下降沿触发
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetTriggerActivation.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerActivation.restype = c_int
    res = MVGigE.MVSetTriggerActivation(c_uint64(hCam), act)
    return res

#MVGetTriggerDelay(HANDLE hCam, uint32_t* pDelay_us)
def MVGetTriggerDelay(hCam):
    """
        读取触发延时
    :param hCam: 相机句柄
    :param pDelay_us: 触发延时,单位us
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTriggerDelay.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTriggerDelay.restype = c_int
    pDelay_us = c_uint()
    res = MVGigE.MVGetTriggerDelay(c_uint64(hCam), byref(pDelay_us))
    return res, pDelay_us.value

#MVGetTriggerDelayRange(HANDLE hCam, uint32_t* pMin, uint32_t* pMax)
def MVGetTriggerDelayRange(hCam):
    """
        读取触发延时范围
    :param hCam: 相机句柄
    :param pMin: 触发延时最小值,单位us
    :param pMax: 触发延时最大值,单位us
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTriggerDelayRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetTriggerDelayRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetTriggerDelayRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value

#MVSetTriggerDelay(HANDLE hCam, uint32_t nDelay_us)
def MVSetTriggerDelay(hCam, nDelay_us):
    """
        设置相机接到触发信号后延迟多少微秒后再开始曝光。
    :param hCam: 相机句柄
    :param nDelay_us: 
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetTriggerDelay.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTriggerDelay.restype = c_int
    res = MVGigE.MVSetTriggerDelay(c_uint64(hCam), c_uint(nDelay_us))
    return res

#MVTriggerSoftware(HANDLE hCam)
def MVTriggerSoftware(hCam):
    """
        发出软件触发指令
    :param hCam: 相机句柄
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVTriggerSoftware.argtype = (c_uint64)
    MVGigE.MVTriggerSoftware.restype = c_int
    res = MVGigE.MVTriggerSoftware(c_uint64(hCam))
    return res

#MVGetStrobeSource(HANDLE hCam, LineSourceEnums* pSource)
def MVGetStrobeSource(hCam):
    """
        读取闪光同步信号源
    :param hCam: 相机句柄
    :param pSource: 闪光同步信号源
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetStrobeSource.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetStrobeSource.restype = c_int
    pSource = c_uint()
    res = MVGigE.MVGetStrobeSource(c_uint64(hCam), byref(pSource))
    return res, pSource.value

#MVSetStrobeSource(HANDLE hCam, LineSourceEnums source)
def MVSetStrobeSource(hCam, source):
    """
        闪光同步信号源
    :param hCam: 
    :param source: 
                    LineSource_Off：关闭闪光同步
                    LineSource_ExposureActive：曝光的同时闪光
                    LineSource_Timer1Active：由定时器控制
                    LineSource_UserOutput0：由用户通过指令控制
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetStrobeSource.argtype = (c_uint64, c_uint)
    MVGigE.MVSetStrobeSource.restype = c_int
    res = MVGigE.MVSetStrobeSource(c_uint64(hCam), source)
    return res

#MVGetStrobeInvert(HANDLE hCam, bool* pInvert)
def MVGetStrobeInvert(hCam):
    """
        读取闪光同步是否反转
    :param hCam: 相机句柄
    :param pInvert: 
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetStrobeInvert.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetStrobeInvert.restype = c_int
    pInvert = c_bool()
    res = MVGigE.MVGetStrobeInvert(c_uint64(hCam), byref(pInvert))
    return res, pInvert.value

#MVSetStrobeInvert(HANDLE hCam, bool bInvert)
def MVSetStrobeInvert(hCam, bInvert):
    """
        闪光同步是否反转，即闪光同步有效时输出高电平还是低电平。
    :param hCam: 相机句柄
    :param bInvert: 
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetStrobeInvert.argtype = (c_uint64, c_bool)
    MVGigE.MVSetStrobeInvert.restype = c_int
    res = MVGigE.MVSetStrobeInvert(c_uint64(hCam), c_bool(bInvert))
    return res

#MVGetUserOutputValue0(HANDLE hCam, bool* pSet)
def MVGetUserOutputValue0(hCam):
    """
        读取用户设置的闪光同步
    :param hCam: 相机句柄
    :param pSet: 
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetUserOutputValue0.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetUserOutputValue0.restype = c_int
    pSet = c_bool()
    res = MVGigE.MVGetUserOutputValue0(c_uint64(hCam), byref(pSet))
    return res, pSet.value

#MVSetUserOutputValue0(HANDLE hCam, bool bSet)
def MVSetUserOutputValue0(hCam, bSet):
    """
        当闪光同步源选为UserOutput时
                主机可以通过MVSetUserOutputValue0来控制闪光同步输出高电平或低电平。
    :param hCam: 相机句柄
    :param bSet: 设置电平
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetUserOutputValue0.argtype = (c_uint64, c_bool)
    MVGigE.MVSetUserOutputValue0.restype = c_int
    res = MVGigE.MVSetUserOutputValue0(c_uint64(hCam), c_bool(bSet))
    return res

#MVSetHeartbeatTimeout(HANDLE hCam, unsigned long nTimeOut);//unit m
def MVSetHeartbeatTimeout(hCam, nTimeOut):
    """
     设置心跳超时时间
    :param hCam: 相机句柄
    :param nTimeOut: 心跳超时时间 单位ms
    :return:    MVST_SUCCESS            : 成功
    :note:  应用程序打开相机后，正常情况下会去关闭相机。但有时程序会意外中断，没有正常关闭相机。
     *          这种情况下相机就无法再次打开了（相机认为还在被原来软件控制着）。 
     *          因此设计了heartbeat，应用程序无论是否在采集图像，每隔一定时间(一般是1000ms)都要去访问一下相机。
     *          当超过HeartbeatTimeout时间没有来自应用程序的访问，相机就认为应用程序中断了，会自动关闭原有连接，
     *          以迎接新的连接。
     *          定时访问相机的操作在SDK内部已经实现。编程时无需再次实现。
     *          在设置断点调试程序时，可以将nTimeOut设置长一些，否则相机会自动关闭连接。
     *          Release状态下设置短一些(3000ms),否则应用程序意外退出时，相机会长时间无法重新打开。
    """
    MVGigE.MVSetHeartbeatTimeout.argtype = (c_uint64, c_ulong)
    MVGigE.MVSetHeartbeatTimeout.restype = c_int
    res = MVGigE.MVSetHeartbeatTimeout(c_uint64(hCam), c_ulong(nTimeOut))
    return res

#MVGetPacketSize(HANDLE hCam, unsigned int* pPacketSize)
def MVGetPacketSize(hCam):
    """
        读取网络数据包大小
    :param hCam: 相机句柄
    :param pPacketSize: 数据包大小
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetPacketSize.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetPacketSize.restype = c_int
    pPacketSize = c_uint()
    res = MVGigE.MVGetPacketSize(c_uint64(hCam), byref(pPacketSize))
    return res, pPacketSize.value

#MVGetPacketSizeRange(HANDLE hCam, unsigned int* pMin, unsigned int* pMax)
def MVGetPacketSizeRange(hCam):
    """
        读取网络数据包大小的范围。
    :param hCam: 相机句柄
    :param pMin: 网络数据包最小值
    :param pMax: 网络数据包最大值
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetPacketSizeRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetPacketSizeRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetPacketSizeRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value

#MVSetPacketSize(HANDLE hCam, unsigned int nPacketSize)
def MVSetPacketSize(hCam, nPacketSize):
    """
        设置网络数据包的大小。
    :param hCam: 相机句柄
    :param nPacketSize: 网络数据包大小(单位:字节)。该大小必须小于网卡能够支持的最大巨型帧(Jumbo Frame)。
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetPacketSize.argtype = (c_uint64, c_uint)
    MVGigE.MVSetPacketSize.restype = c_int
    res = MVGigE.MVSetPacketSize(c_uint64(hCam), c_uint(nPacketSize))
    return res

#MVGetPacketDelay(HANDLE hCam, unsigned int* pDelay_us)
def MVGetPacketDelay(hCam):
    """
        读取网络数据包间隔。
    :param hCam: 相机句柄
    :param pDelay_us: 数据包间隔时间，单位us
    :return:  MVST_SUCCESS              : 成功
    """
    MVGigE.MVGetPacketDelay.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetPacketDelay.restype = c_int
    pDelay_us = c_uint()
    res = MVGigE.MVGetPacketDelay(c_uint64(hCam), byref(pDelay_us))
    return res, pDelay_us.value

#MVGetPacketDelayRange(HANDLE hCam, unsigned int* pMin, unsigned int* pMax)
def MVGetPacketDelayRange(hCam):
    """
        读取网络数据包间隔范围
    :param hCam: 相机句柄
    :param pMin: 数据包间隔时间最小值，单位us
    :param pMax: 数据包间隔时间最大值，单位us
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetPacketDelayRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetPacketDelayRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetPacketDelayRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value

#MVSetPacketDelay(HANDLE hCam, unsigned int nDelay_us)
def MVSetPacketDelay(hCam, nDelay_us):
    """
        设置网络数据包之间的时间间隔。如果网卡或电脑的性能欠佳，无法处理高速到达的数据包，会导致丢失数据包，
                从而使图像不完整。可以通过增加数据包之间的时间间隔以保证图像传输。但是增加该值将增加图像的时间延迟，
                并有可能影像到帧率。
    :param hCam: 
    :param nDelay_us: 时间间隔(单位:微秒)
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetPacketDelay.argtype = (c_uint64, c_uint)
    MVGigE.MVSetPacketDelay.restype = c_int
    res = MVGigE.MVSetPacketDelay(c_uint64(hCam), c_uint(nDelay_us))
    return res

#MVGetTimerDelay(HANDLE hCam, uint32_t* pDelay)
def MVGetTimerDelay(hCam):
    """
        读取定时器延时
    :param hCam: 相机句柄
    :param pDelay: 定时器延时
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTimerDelay.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTimerDelay.restype = c_int
    pDelay = c_uint()
    res = MVGigE.MVGetTimerDelay(c_uint64(hCam), byref(pDelay))
    return res, pDelay.value

#MVGetTimerDelayRange(HANDLE hCam, uint32_t* pMin, uint32_t* pMax)
def MVGetTimerDelayRange(hCam):
    """
        读取定时器延时的范围
    :param hCam: 相机句柄
    :param pMin: 定时器延时的最小值
    :param pMax: 定时器延时的最大值
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTimerDelayRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetTimerDelayRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetTimerDelayRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value

#MVSetTimerDelay(HANDLE hCam, uint32_t nDelay)
def MVSetTimerDelay(hCam, nDelay):
    """
        当闪光同步源选为Timer1时MVSetStobeSource(hCam,LineSource_Timer1Active)
                设置Timer1在接到触发信号后延迟多少us开始计时
    :param hCam: 相机句柄
    :param nDelay: 接到触发信号后延迟多少us开始计时
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetTimerDelay.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTimerDelay.restype = c_int
    res = MVGigE.MVSetTimerDelay(c_uint64(hCam), c_uint(nDelay))
    return res

#MVGetTimerDuration(HANDLE hCam, uint32_t* pDuration)
def MVGetTimerDuration(hCam):
    """
        读取定时器计时时长
    :param hCam: 相机句柄
    :param pDuration: 定时器计时时长
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTimerDuration.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetTimerDuration.restype = c_int
    pDuration = c_uint()
    res = MVGigE.MVGetTimerDuration(c_uint64(hCam), byref(pDuration))
    return res, pDuration.value

#MVGetTimerDurationRange(HANDLE hCam, uint32_t* pMin, uint32_t* pMax)
def MVGetTimerDurationRange(hCam):
    """
        读取定时器计时时长取值范围
    :param hCam: 相机句柄
    :param pMin: 定时器计时时长最小值
    :param pMax: 定时器计时时长最大值
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVGetTimerDurationRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetTimerDurationRange.restype = c_int
    pMin = c_uint()
    pMax = c_uint()
    res = MVGigE.MVGetTimerDurationRange(c_uint64(hCam), byref(pMin), byref(pMax))
    return res, pMin.value, pMax.value

#MVSetTimerDuration(HANDLE hCam, uint32_t nDuration)
def MVSetTimerDuration(hCam, nDuration):
    """
        当闪光同步源选为Timer1时MVSetStobeSource(hCam,LineSource_Timer1Active)
                设置Timer1在开始计时后，计时多长时间。
    :param hCam: 
    :param nDuration: 设置Timer1在开始计时后，计时多长时间(us)。即输出高/低电平的脉冲宽度。
    :return:  MVST_SUCCESS          : 成功
    """
    MVGigE.MVSetTimerDuration.argtype = (c_uint64, c_uint)
    MVGigE.MVSetTimerDuration.restype = c_int
    res = MVGigE.MVSetTimerDuration(c_uint64(hCam), c_uint(nDuration))
    return res

#MVBayerToBGR(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat,bool bMultiCores=FALSE)
def MVBayerToBGR(hCam, psrc, dststep, width, height, pixelformat, bMultiCores=FALSE):
    """
     将Bayer格式的8bit单通道图转换为BGR格式的8Bit三通道图
    :param hCam: 相机句柄
    :param psrc: 单通道图像的指针
    :param pdst: 三通道图像指针
    :param dststep: 三通道图像一行图像的字节数。通常为图像宽度*3，但是会为了4字节对齐会补几个字节。
    :param width: 图像宽度
    :param height: 图像高度
    :param pixelformat: 像素格式，由MVGetPixelFormat取得
    :param bMultiCores: 是否使用CPU多核计算
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVBayerToBGR.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint, c_bool)
    MVGigE.MVBayerToBGR.restype = c_int
    pdst = (c_ubyte * 3 * width * height)()
    res = MVGigE.MVBayerToBGR(c_uint64(hCam), psrc, byref(pdst), c_uint(dststep), c_uint(width), c_uint(height), pixelformat, c_bool(bMultiCores))
    image = np.ctypeslib.as_array(pdst)
    return res, image

#MVBayerToBGR16(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat )
def MVBayerToBGR16(hCam, psrc, dststep, width, height, pixelformat):
    """
     将Bayer格式的16bit单通道图转换为BGR格式的16Bit三通道图
    :param hCam: 相机句柄
    :param psrc: 单通道图像的指针
    :param dststep: 三通道图像一行图像的字节数。
    :param width: 图像宽度
    :param height: 图像高度
    :param pixelformat: 像素格式，由MVGetPixelFormat取得
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVBayerToBGR16.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint)
    MVGigE.MVBayerToBGR16.restype = c_int
    pdst = (c_ushort * 3 * width * height)()
    res = MVGigE.MVBayerToBGR16(c_uint64(hCam), psrc, byref(pdst), c_uint(dststep), c_uint(width), c_uint(height), pixelformat)
    image = np.ctypeslib.as_array(pdst)
    return res, image

#MVBayerToRGB(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat,bool bMultiCores=FALSE)
def MVBayerToRGB(hCam, psrc, dststep, width, height, pixelformat, bMultiCores=FALSE):
    """
     将Bayer格式的8bit单通道图转换为RGB格式的8Bit三通道图
    :param hCam: 相机句柄
    :param psrc: 单通道图像的指针
    :param dststep: 三通道图像一行图像的字节数。通常为图像宽度*3，但是会为了4字节对齐会补几个字节。
    :param width: 图像宽度
    :param height: 图像高度
    :param pixelformat: 像素格式，由MVGetPixelFormat取得
    :param bMultiCores: 是否使用CPU多核计算
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVBayerToRGB.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint, c_bool)
    MVGigE.MVBayerToRGB.restype = c_int
    pdst = (c_ubyte * 3 * width * height)()
    res = MVGigE.MVBayerToRGB(c_uint64(hCam), psrc, byref(pdst), c_uint(dststep), c_uint(width), c_uint(height), pixelformat, c_bool(bMultiCores))
    image = np.ctypeslib.as_array(pdst)
    return res, image

#MVBayerToRGB16(HANDLE hCam, void *psrc,void *pdst,unsigned int dststep,unsigned int width,unsigned int height,MV_PixelFormatEnums pixelformat )
def MVBayerToRGB16(hCam, psrc, dststep, width, height, pixelformat):
    """
     将Bayer格式的16bit单通道图转换为RGB格式的16Bit三通道图
    :param hCam: 相机句柄
    :param psrc: 单通道图像的指针
    :param pdst: 三通道图像指针
    :param dststep: 三通道图像一行图像的字节数。
    :param width: 图像宽度
    :param height: 图像高度
    :param pixelformat: 像素格式，由MVGetPixelFormat取得
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVBayerToRGB16.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint, c_uint, c_uint)
    MVGigE.MVBayerToRGB16.restype = c_int
    pdst = (c_ushort * 3 * width * height)()
    res = MVGigE.MVBayerToRGB16(c_uint64(hCam), psrc, byref(pdst), c_uint(dststep), c_uint(width), c_uint(height), pixelformat)
    image = np.ctypeslib.as_array(pdst)
    return res, image

#MVImageBayerToBGR(HANDLE hCam, MV_IMAGE_INFO* pInfo, MVImage* pImage)
def MVImageBayerToBGR(hCam, pInfo):
    """
     将Bayer格式的8bit单通道图转换为BGR格式的8Bit三通道图
    :param hCam: 相机句柄
    :param pInfo: 采集Callback函数中传来的图像信息指针
    :param pImage: 转换结果图像的指针
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVImageBayerToBGR.argtype = (c_uint64, POINTER(MV_IMAGE_INFO), c_void_p)
    MVGigE.MVImageBayerToBGR.restype = c_int
    pImage = MVImage()
    res = MVGigE.MVImageBayerToBGR(c_uint64(hCam), pInfo, byref(pImage))
    return res, pImage

#MVInfo2Image(HANDLE hCam, MV_IMAGE_INFO* pInfo, MVImage* pImage)
def MVInfo2Image(hCam, pInfo):
    """
     将回调函数收到的图像信息转换为图像。
    :param hCam: 相机句柄
    :param pInfo: 采集Callback函数中传来的图像信息指针
    :param pImage: 转换结果图像的指针
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVInfo2Image.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVInfo2Image.restype = c_int
    pInfo = c_uint()
    pImage = c_uint()
    res = MVGigE.MVInfo2Image(c_uint64(hCam), byref(pInfo), byref(pImage))
    return res, pInfo.value, pImage.value

#MVImageBayerToBGREx( HANDLE hCam,MV_IMAGE_INFO *pInfo,MVImage *pImage,double fGamma,bool bColorCorrect,int nContrast )
def MVImageBayerToBGREx(hCam, fGamma, bColorCorrect, nContrast):
    """
     将Bayer格式的8bit单通道图转换为BGR格式的8Bit三通道图,同时调整图像的GAMMA,颜色和反差
    :param hCam: 相机句柄
    :param pInfo: 采集Callback函数中传来的图像信息指针
    :param pImage: 转换结果图像的指针
    :param fGamma: Gamma校正值，1为不校正，<1时，将暗部提升。
    :param bColorCorrect: 是否进行颜色校正，进行颜色校正后，图像会变得更鲜艳。
    :param nContrast: 是否调整反差，范围为0－50，当该值大于0时，图像反差会更强。
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVImageBayerToBGREx.argtype = (c_uint64, c_void_p, c_void_p, c_double, c_bool, c_int)
    MVGigE.MVImageBayerToBGREx.restype = c_int
    pInfo = c_uint()
    pImage = c_uint()
    res = MVGigE.MVImageBayerToBGREx(c_uint64(hCam), byref(pInfo), byref(pImage), c_double(fGamma), c_bool(bColorCorrect), c_int(nContrast))
    return res, pInfo.value, pImage.value

#MVZoomImageBGR(HANDLE hCam, unsigned char* pSrc, int srcWidth, int srcHeight, unsigned char* pDst, double fFactorX, double fFactorY)
def MVZoomImageBGR(hCam, pSrc, srcWidth, srcHeight, fFactorX, fFactorY):
    """
     BGR格式三通道图像缩放
    :param hCam: 相机句柄
    :param pSrc: 源图像指针
    :param srcWidth: 源图像宽度
    :param srcHeight: 源图像高度
    :param pDst: 缩放后图像指针
    :param fFactorX: 缩放比例
    :param fFactorY: 缩放比例
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVZoomImageBGR.argtype = (c_uint64, c_void_p, c_int, c_int, c_void_p, c_double, c_double)
    MVGigE.MVZoomImageBGR.restype = c_int
    zoomX  = int(srcWidth * 3 * fFactorX) 
    zoomY = int(srcHeight * fFactorY)
    pDst = (c_ubyte * zoomX * zoomY)()
    res = MVGigE.MVZoomImageBGR(c_uint64(hCam), pSrc, c_int(srcWidth), c_int(srcHeight), byref(pDst), c_double(fFactorX), c_double(fFactorY))
    return res, pDst

#MVGetStreamStatistic(HANDLE hCam, MVStreamStatistic* pStatistic)
def MVGetStreamStatistic(hCam):
    """
        获取数据传输的统计信息
    :param hCam: 相机句柄
    :param pStatistic: 统计信息
    :return:    JMVST_SUCCESS           : 成功
    """
    MVGigE.MVGetStreamStatistic.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetStreamStatistic.restype = c_int
    pStatistic = MVStreamStatistic()
    res = MVGigE.MVGetStreamStatistic(c_uint64(hCam), byref(pStatistic))
    return res, pStatistic

#MVLoadUserSet(HANDLE hCam, UserSetSelectorEnums userset)
def MVLoadUserSet(hCam, userset):
    """
        读取并应用某组用户预设的参数
    :param hCam: 相机句柄
    :param userset: 
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVLoadUserSet.argtype = (c_uint64, c_uint)
    MVGigE.MVLoadUserSet.restype = c_int
    res = MVGigE.MVLoadUserSet(c_uint64(hCam), userset)
    return res

#MVSaveUserSet(HANDLE hCam, UserSetSelectorEnums userset)
def MVSaveUserSet(hCam, userset):
    """
        将当前相机的参数保存到用户设置中。
    :param hCam: 相机句柄
    :param userset: 
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSaveUserSet.argtype = (c_uint64, c_uint)
    MVGigE.MVSaveUserSet.restype = c_int
    res = MVGigE.MVSaveUserSet(c_uint64(hCam), userset)
    return res

#MVSetDefaultUserSet(HANDLE hCam, UserSetSelectorEnums userset)
def MVSetDefaultUserSet(hCam, userset):
    """
        设置相机上电开机时默认读取并应用哪一组用户设置
    :param hCam: 相机句柄
    :param userset: 
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetDefaultUserSet.argtype = (c_uint64, c_uint)
    MVGigE.MVSetDefaultUserSet.restype = c_int
    res = MVGigE.MVSetDefaultUserSet(c_uint64(hCam), userset)
    return res

#MVGetDefaultUserSet(HANDLE hCam, UserSetSelectorEnums* pUserset)
def MVGetDefaultUserSet(hCam):
    """
        读取相机上电开机时默认读取并应用哪一组用户设置
    :param hCam: 相机句柄
    :param pUserset: 用户设置
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetDefaultUserSet.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetDefaultUserSet.restype = c_int
    pUserset = c_uint()
    res = MVGigE.MVGetDefaultUserSet(c_uint64(hCam), byref(pUserset))
    return res, pUserset.value

#MVImageFlip(HANDLE hCam, MVImage* pSrcImage, MVImage* pDstImage, ImageFlipType flipType)
def MVImageFlip(hCam, flipType):
    """
     图像翻转
    :param hCam: 相机句柄
    :param pSrcImage: 源图像指针
    :param pDstImage: 结果图像指针。如果为NULL，则翻转的结果还在源图像内。
    :param flipType: 翻转类型。FlipHorizontal:左右翻转,FlipVertical:上下翻转,FlipBoth:旋转180度
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVImageFlip.argtype = (c_uint64, c_void_p, c_void_p, ImageFlipType)
    MVGigE.MVImageFlip.restype = c_int
    pSrcImage = c_uint()
    pDstImage = c_uint()
    res = MVGigE.MVImageFlip(c_uint64(hCam), byref(pSrcImage), byref(pDstImage), flipType)
    return res, pSrcImage.value, pDstImage.value

#MVImageRotate(HANDLE hCam, MVImage* pSrcImage, MVImage* pDstImage, ImageRotateType roateType)
def MVImageRotate(hCam, roateType):
    """
     图像旋转
    :param hCam: 相机句柄
    :param pSrcImage: 源图像指针
    :param pDstImage: 结果图像指针,不能为NULL。结果图像的宽度和高度应该和源图像的宽度和高度互换。
    :param roateType: 旋转类型：Rotate90DegCw, Rotate90DegCcw
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVImageRotate.argtype = (c_uint64, c_void_p, c_void_p, ImageRotateType)
    MVGigE.MVImageRotate.restype = c_int
    pSrcImage = c_uint()
    pDstImage = c_uint()
    res = MVGigE.MVImageRotate(c_uint64(hCam), byref(pSrcImage), byref(pDstImage), roateType)
    return res, pSrcImage.value, pDstImage.value

#MVBGRToGray(HANDLE hCam, unsigned char* psrc, unsigned char* pdst, unsigned int width, unsigned int height)
def MVBGRToGray(hCam, psrc, width, height):
    """
     将彩色BGR三通道24bit图像转换为灰度单通道8bit图像
    :param hCam: 相机句柄
    :param psrc: 彩色BGR三通道24bit图像指针
    :param pdst: 灰度单通道8bit图像指针
    :param width: 图像宽度
    :param height: 图像高度
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVBGRToGray.argtype = (c_uint64, c_void_p, c_void_p, c_uint, c_uint)
    MVGigE.MVBGRToGray.restype = c_int
    pdst = (c_ubyte * width * height)()
    res = MVGigE.MVBGRToGray(c_uint64(hCam), psrc, byref(pdst), c_uint(width), c_uint(height))
    return res, pdst

#MVImageBGRToGray(HANDLE hCam, MVImage* pSrcImage, MVImage* pDstImage)
def MVImageBGRToGray(hCam):
    """
     将彩色BGR三通道24bit图像转换为灰度单通道8bit图像
    :param hCam: 相机句柄
    :param pSrcImage: 彩色BGR三通道24bit图像指针
    :param pDstImage: 灰度单通道8bit图像指针。宽度高度必须和源图相同
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVImageBGRToGray.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVImageBGRToGray.restype = c_int
    pSrcImage = c_uint()
    pDstImage = c_uint()
    res = MVGigE.MVImageBGRToGray(c_uint64(hCam), byref(pSrcImage), byref(pDstImage))
    return res, pSrcImage.value, pDstImage.value

#MVImageBGRToYUV(HANDLE hCam, MVImage* pSrcImage, unsigned char* pDst)
def MVImageBGRToYUV(hCam):
    """
        将彩色BGR三通道24bit图像转换为YUV图像 
    :param hCam: 相机句柄
    :param pSrcImage: 彩色BGR三通道24bit图像指针
    :param pDst: YUV图像指针 (YUV422)
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVImageBGRToYUV.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVImageBGRToYUV.restype = c_int
    pSrcImage = c_uint()
    pDst = c_ubyte()
    res = MVGigE.MVImageBGRToYUV(c_uint64(hCam), byref(pSrcImage), byref(pDst))
    return res, pSrcImage.value, pDst.value

#MVGrayToBGR(HANDLE hCam, unsigned char* pSrc, unsigned char* pDst, int width, int height)
def MVGrayToBGR(hCam, pSrc, width, height):
    """
     将灰度单通道8bit图像转换为彩色BGR三通道24bit图像。转换后三个通道的值是相同的。
    :param hCam: 相机句柄
    :param pSrc: 灰度单通道8bit图像指针
    :param pDst: 彩色BGR三通道24bit图像指针
    :param width: 图像宽度
    :param height: 图像高度
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGrayToBGR.argtype = (c_uint64, c_void_p, c_void_p, c_int, c_int)
    MVGigE.MVGrayToBGR.restype = c_int
    pDst = (c_ubyte * width * height * 3)()
    res = MVGigE.MVGrayToBGR(c_uint64(hCam), pSrc, byref(pDst), c_int(width), c_int(height))
    return res, pDst

#MVGetExposureAuto(HANDLE hCam, ExposureAutoEnums* pExposureAuto)
def MVGetExposureAuto(hCam):
    """
     获取当前自动曝光模式
    :param hCam: 相机句柄
    :param pExposureAuto: 当前自动曝光模式
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetExposureAuto.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetExposureAuto.restype = c_int
    pExposureAuto = c_uint()
    res = MVGigE.MVGetExposureAuto(c_uint64(hCam), byref(pExposureAuto))
    return res, pExposureAuto.value

#MVSetExposureAuto(HANDLE hCam, ExposureAutoEnums ExposureAuto)
def MVSetExposureAuto(hCam, ExposureAuto):
    """
     设置自动曝光模式
    :param hCam: 相机句柄
    :param ExposureAuto: 自动曝光模式
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetExposureAuto.argtype = (c_uint64, c_uint)
    MVGigE.MVSetExposureAuto.restype = c_int
    res = MVGigE.MVSetExposureAuto(c_uint64(hCam), ExposureAuto)
    return res

#MVGetGainAuto(HANDLE hCam, GainAutoEnums* pGainAuto)
def MVGetGainAuto(hCam):
    """
     获取当前自动增益模式
    :param hCam: 相机句柄
    :param pGainAuto: 当前自动增益模式的
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetGainAuto.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGainAuto.restype = c_int
    pGainAuto = c_uint()
    res = MVGigE.MVGetGainAuto(c_uint64(hCam), byref(pGainAuto))
    return res, pGainAuto.value

#MVSetGainAuto(HANDLE hCam, GainAutoEnums GainAuto)
def MVSetGainAuto(hCam, GainAuto):
    """
     设置当前自动增益模式
    :param hCam: 相机句柄
    :param GainAuto: 自动增益模式
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetGainAuto.argtype = (c_uint64, c_uint)
    MVGigE.MVSetGainAuto.restype = c_int
    res = MVGigE.MVSetGainAuto(c_uint64(hCam), GainAuto)
    return res

#MVGetBalanceWhiteAuto(HANDLE hCam, BalanceWhiteAutoEnums* pBalanceWhiteAuto)
def MVGetBalanceWhiteAuto(hCam):
    """
     获取当前自动白平衡模式
    :param hCam: 相机句柄
    :param pBalanceWhiteAuto: 当前自动白平衡模式
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetBalanceWhiteAuto.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetBalanceWhiteAuto.restype = c_int
    pBalanceWhiteAuto = c_uint()
    res = MVGigE.MVGetBalanceWhiteAuto(c_uint64(hCam), byref(pBalanceWhiteAuto))
    return res, pBalanceWhiteAuto.value

#MVSetBalanceWhiteAuto(HANDLE hCam, BalanceWhiteAutoEnums BalanceWhiteAuto)
def MVSetBalanceWhiteAuto(hCam, BalanceWhiteAuto):
    """
     设置自动白平衡模式
    :param hCam: 相机句柄
    :param BalanceWhiteAuto: 自动白平衡模式
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetBalanceWhiteAuto.argtype = (c_uint64, c_uint)
    MVGigE.MVSetBalanceWhiteAuto.restype = c_int
    res = MVGigE.MVSetBalanceWhiteAuto(c_uint64(hCam), BalanceWhiteAuto)
    return res

#MVGetAutoGainLowerLimit(HANDLE hCam, double* pAutoGainLowerLimit)
def MVGetAutoGainLowerLimit(hCam):
    """
     获取自动调整增益时，增益调整范围的最小值
    :param hCam: 相机句柄
    :param pAutoGainLowerLimit: 增益调整范围的最小值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetAutoGainLowerLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoGainLowerLimit.restype = c_int
    pAutoGainLowerLimit = c_double()
    res = MVGigE.MVGetAutoGainLowerLimit(c_uint64(hCam), byref(pAutoGainLowerLimit))
    return res, pAutoGainLowerLimit.value

#MVSetAutoGainLowerLimit(HANDLE hCam, double fAutoGainLowerLimit)
def MVSetAutoGainLowerLimit(hCam, fAutoGainLowerLimit):
    """
     设置自动调整增益时，增益调整范围的最小值
    :param hCam: 相机句柄
    :param fAutoGainLowerLimit: 增益调整范围的最小值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetAutoGainLowerLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoGainLowerLimit.restype = c_int
    res = MVGigE.MVSetAutoGainLowerLimit(c_uint64(hCam), c_double(fAutoGainLowerLimit))
    return res

#MVGetAutoGainUpperLimit(HANDLE hCam, double* pAutoGainUpperLimit)
def MVGetAutoGainUpperLimit(hCam):
    """
     获取自动调整增益时，增益调整范围的最大值
    :param hCam: 相机句柄
    :param pAutoGainUpperLimit: 增益调整范围的最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetAutoGainUpperLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoGainUpperLimit.restype = c_int
    pAutoGainUpperLimit = c_double()
    res = MVGigE.MVGetAutoGainUpperLimit(c_uint64(hCam), byref(pAutoGainUpperLimit))
    return res, pAutoGainUpperLimit.value

#MVSetAutoGainUpperLimit(HANDLE hCam, double fAutoGainUpperLimit)
def MVSetAutoGainUpperLimit(hCam, fAutoGainUpperLimit):
    """
     设置自动调整增益时，增益调整范围的最大值
    :param hCam: 相机句柄
    :param fAutoGainUpperLimit: 曝光时间调整范围的最小值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetAutoGainUpperLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoGainUpperLimit.restype = c_int
    res = MVGigE.MVSetAutoGainUpperLimit(c_uint64(hCam), c_double(fAutoGainUpperLimit))
    return res

#MVGetAutoExposureTimeLowerLimit(HANDLE hCam, double* pAutoExposureTimeLowerLimit)
def MVGetAutoExposureTimeLowerLimit(hCam):
    """
     获取自动调整曝光时间时，曝光时间调整范围的最小值
    :param hCam: 相机句柄
    :param pAutoExposureTimeLowerLimit: 曝光时间调整范围的最小值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetAutoExposureTimeLowerLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoExposureTimeLowerLimit.restype = c_int
    pAutoExposureTimeLowerLimit = c_double()
    res = MVGigE.MVGetAutoExposureTimeLowerLimit(c_uint64(hCam), byref(pAutoExposureTimeLowerLimit))
    return res, pAutoExposureTimeLowerLimit.value

#MVSetAutoExposureTimeLowerLimit(HANDLE hCam, double fAutoExposureTimeLowerLimit)
def MVSetAutoExposureTimeLowerLimit(hCam, fAutoExposureTimeLowerLimit):
    """
     设置自动调整曝光时间时，曝光时间调整范围的最小值
    :param hCam: 相机句柄
    :param fAutoExposureTimeLowerLimit: 曝光时间调整范围的最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetAutoExposureTimeLowerLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoExposureTimeLowerLimit.restype = c_int
    res = MVGigE.MVSetAutoExposureTimeLowerLimit(c_uint64(hCam), c_double(fAutoExposureTimeLowerLimit))
    return res

#MVGetAutoExposureTimeUpperLimit(HANDLE hCam, double* pAutoExposureTimeUpperLimit)
def MVGetAutoExposureTimeUpperLimit(hCam):
    """
     获取自动调整曝光时间时，曝光时间调整范围的最大值
    :param hCam: 相机句柄
    :param pAutoExposureTimeUpperLimit: 曝光时间调整范围的最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetAutoExposureTimeUpperLimit.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoExposureTimeUpperLimit.restype = c_int
    pAutoExposureTimeUpperLimit = c_double()
    res = MVGigE.MVGetAutoExposureTimeUpperLimit(c_uint64(hCam), byref(pAutoExposureTimeUpperLimit))
    return res, pAutoExposureTimeUpperLimit.value

#MVSetAutoExposureTimeUpperLimit(HANDLE hCam, double fAutoExposureTimeUpperLimit)
def MVSetAutoExposureTimeUpperLimit(hCam, fAutoExposureTimeUpperLimit):
    """
     设置自动调整曝光时间时，曝光时间调整范围的最大值
    :param hCam: 相机句柄
    :param fAutoExposureTimeUpperLimit: 
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetAutoExposureTimeUpperLimit.argtype = (c_uint64, c_double)
    MVGigE.MVSetAutoExposureTimeUpperLimit.restype = c_int
    res = MVGigE.MVSetAutoExposureTimeUpperLimit(c_uint64(hCam), c_double(fAutoExposureTimeUpperLimit))
    return res

#MVGetAutoTargetValue(HANDLE hCam, int* pAutoTargetValue)
def MVGetAutoTargetValue(hCam):
    """
     获取自动调整亮度(曝光、增益)时，期望调整到的图像亮度
    :param hCam: 相机句柄
    :param pAutoTargetValue: 期望调整到的图像亮度
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetAutoTargetValue.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoTargetValue.restype = c_int
    pAutoTargetValue = c_int()
    res = MVGigE.MVGetAutoTargetValue(c_uint64(hCam), byref(pAutoTargetValue))
    return res, pAutoTargetValue.value

#MVSetAutoTargetValue(HANDLE hCam, int nAutoTargetValue)
def MVSetAutoTargetValue(hCam, nAutoTargetValue):
    """
     设置自动调整亮度(曝光、增益)时，期望调整到的图像亮度
    :param hCam: 相机句柄
    :param nAutoTargetValue: 期望调整到的图像亮度
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetAutoTargetValue.argtype = (c_uint64, c_int)
    MVGigE.MVSetAutoTargetValue.restype = c_int
    res = MVGigE.MVSetAutoTargetValue(c_uint64(hCam), c_int(nAutoTargetValue))
    return res

#MVGetAutoFunctionProfile(HANDLE hCam, AutoFunctionProfileEnums* pAutoFunctionProfile)
def MVGetAutoFunctionProfile(hCam):
    """
     当自动增益和自动曝光时间都打开时，获取哪一个值优先调整
    :param hCam: 相机句柄
    :param pAutoFunctionProfile: 增益优先或曝光时间优先
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetAutoFunctionProfile.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoFunctionProfile.restype = c_int
    pAutoFunctionProfile = c_uint()
    res = MVGigE.MVGetAutoFunctionProfile(c_uint64(hCam), byref(pAutoFunctionProfile))
    return res, pAutoFunctionProfile.value

#MVSetAutoFunctionProfile(HANDLE hCam, AutoFunctionProfileEnums AutoFunctionProfile)
def MVSetAutoFunctionProfile(hCam, AutoFunctionProfile):
    """
     当自动增益和自动曝光时间都打开时，设置哪一个值优先调整
    :param hCam: 相机句柄
    :param AutoFunctionProfile: 增益优先或曝光时间优先
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetAutoFunctionProfile.argtype = (c_uint64, c_uint)
    MVGigE.MVSetAutoFunctionProfile.restype = c_int
    res = MVGigE.MVSetAutoFunctionProfile(c_uint64(hCam), AutoFunctionProfile)
    return res

#MVGetAutoThreshold(HANDLE hCam, int* pAutoThreshold)
def MVGetAutoThreshold(hCam):
    """
        自动增益或自动曝光时，图像亮度与目标亮度差异的容差。
    :param hCam: 相机句柄
    :param pAutoThreshold: 图像亮度与目标亮度差异的容差
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetAutoThreshold.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetAutoThreshold.restype = c_int
    pAutoThreshold = c_int()
    res = MVGigE.MVGetAutoThreshold(c_uint64(hCam), byref(pAutoThreshold))
    return res, pAutoThreshold.value

#MVSetAutoThreshold(HANDLE hCam, int nAutoThreshold)
def MVSetAutoThreshold(hCam, nAutoThreshold):
    """
     自动增益或自动曝光时，图像亮度与目标亮度差异的容差。
    :param hCam: 相机句柄
    :param nAutoThreshold: 图像亮度与目标亮度差异的容差
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetAutoThreshold.argtype = (c_uint64, c_int)
    MVGigE.MVSetAutoThreshold.restype = c_int
    res = MVGigE.MVSetAutoThreshold(c_uint64(hCam), c_int(nAutoThreshold))
    return res

#MVGetGamma(HANDLE hCam, double* pGamma)
def MVGetGamma(hCam):
    """
     获取当前伽马值
    :param hCam: 相机句柄
    :param pGamma: 当前伽马值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetGamma.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetGamma.restype = c_int
    pGamma = c_double()
    res = MVGigE.MVGetGamma(c_uint64(hCam), byref(pGamma))
    return res, pGamma.value

#MVGetGammaRange(HANDLE hCam, double* pGammaMin, double* pGammaMax)
def MVGetGammaRange(hCam):
    """
     获取伽马值可设置的范围
    :param hCam: 相机句柄
    :param pGammaMin: 伽马最小值
    :param pGammaMax: 伽马最大值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetGammaRange.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVGetGammaRange.restype = c_int
    pGammaMin = c_double()
    pGammaMax = c_double()
    res = MVGigE.MVGetGammaRange(c_uint64(hCam), byref(pGammaMin), byref(pGammaMax))
    return res, pGammaMin.value, pGammaMax.value

#MVSetGamma(HANDLE hCam, double fGamma)
def MVSetGamma(hCam, fGamma):
    """
     设置伽马值
    :param hCam: 相机句柄
    :param fGamma: 伽马值
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetGamma.argtype = (c_uint64, c_double)
    MVGigE.MVSetGamma.restype = c_int
    res = MVGigE.MVSetGamma(c_uint64(hCam), c_double(fGamma))
    return res

#MVSetLUT(HANDLE hCam, unsigned long* pLUT, int nCnt)
def MVSetLUT(hCam, nCnt):
    """
     设置查找表
    :param hCam: 相机句柄
    :param pLUT: 查找表数组，unsigned long pLUT[1024];
    :param nCnt: 查找表数组单元个数，必须是1024
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetLUT.argtype = (c_uint64, c_void_p, c_int)
    MVGigE.MVSetLUT.restype = c_int
    pLUT = c_ulong()
    res = MVGigE.MVSetLUT(c_uint64(hCam), byref(pLUT), c_int(nCnt))
    return res, pLUT.value

#MVSetEnableLUT(HANDLE hCam, bool bEnable)
def MVSetEnableLUT(hCam, bEnable):
    """
    　使用查找表
    :param hCam: 相机句柄
    :param bEnable: 是否允许
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetEnableLUT.argtype = (c_uint64, c_bool)
    MVGigE.MVSetEnableLUT.restype = c_int
    res = MVGigE.MVSetEnableLUT(c_uint64(hCam), c_bool(bEnable))
    return res

#MVGetEnableLUT(HANDLE hCam, bool* bEnable)
def MVGetEnableLUT(hCam):
    """
     获取当前是否使用查找表状态
    :param hCam: 
    :param bEnable: 当前是否使用查找表状态
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetEnableLUT.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetEnableLUT.restype = c_int
    bEnable = c_bool()
    res = MVGigE.MVGetEnableLUT(c_uint64(hCam), byref(bEnable))
    return res, bEnable.value

#MVSingleGrab(HANDLE hCam, HANDLE hImage, unsigned long nWaitMs)
def MVSingleGrab(hCam, hImage, nWaitMs):
    """
     采集一帧图像。
    :param hCam: 相机句柄
    :param hImage: 图像句柄。保存采集到的图像。
    :param nWaitMs: 等待多长时间，单位ms
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSingleGrab.argtype = (c_uint64, c_uint64, c_ulong)
    MVGigE.MVSingleGrab.restype = c_int
    res = MVGigE.MVSingleGrab(c_uint64(hCam), c_uint64(hImage), c_ulong(nWaitMs))
    return res

#MVStartGrabWindow(HANDLE hCam, HWND hWnd, HWND hWndMsg)
def MVStartGrabWindow(hCam, hWnd = 0, hWndMsg = 0):
    """
     开始采集，并将采集到的图像显示到指定窗口
    :param hCam: 相机句柄
    :param hWnd: 窗口句柄
    :param hWndMsg: 消息句柄，如果不为NULL,当新的图像采集完毕,会发送消息(WM_USER+0x0200)到此窗口
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVStartGrabWindow.argtype = (c_uint64, c_uint64, c_uint64)
    MVGigE.MVStartGrabWindow.restype = c_int
    res = MVGigE.MVStartGrabWindow(c_uint64(hCam), c_uint64(hWnd), c_uint64(hWndMsg))
    return res

#MVStopGrabWindow(HANDLE hCam)
def MVStopGrabWindow(hCam):
    """
        停止采集到窗口
    :param hCam: 相机句柄
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVStopGrabWindow.argtype = (c_uint64)
    MVGigE.MVStopGrabWindow.restype = c_int
    res = MVGigE.MVStopGrabWindow(c_uint64(hCam))
    return res

#MVFreezeGrabWindow(HANDLE hCam, bool bFreeze)
def MVFreezeGrabWindow(hCam, bFreeze):
    """
        当采集到窗口时，暂停或继续采集。
    :param hCam: 相机句柄
    :param bFreeze: 
     *              TRUE:暂停采集，暂停后可以调用GetSampleGrab函数得到当前图像。
     *              FALSE:继续采集
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVFreezeGrabWindow.argtype = (c_uint64, c_bool)
    MVGigE.MVFreezeGrabWindow.restype = c_int
    res = MVGigE.MVFreezeGrabWindow(c_uint64(hCam), c_bool(bFreeze))
    return res

#MVSetGrabWindow(HANDLE hCam, int xDest, int yDest, int wDest, int hDest, int xSrc, int ySrc, int wSrc, int hSrc)
def MVSetGrabWindow(hCam, xDest, yDest, wDest, hDest, xSrc, ySrc, wSrc, hSrc):
    """
     当采集到窗口时，设置图像显示的区域和比例。
     *  将图像中(xSrc,ySrc,wSrc,hSrc)指定的区域显示到窗口中指定区域(xDest,yDest,wDest,hDest)
    :param hCam: 相机句柄
    :param xDest: 指定显示窗口中目标矩形左上角的逻辑X坐标
    :param yDest: 指定显示窗口中目标矩形左上角的逻辑Y坐标
    :param wDest: 指定显示窗口中目标矩形的宽度
    :param hDest: 指定显示窗口中目标矩形的高度
    :param xSrc: 指定图像源位图左上角的逻辑X坐标
    :param ySrc: 指定图像源位图左上角的逻辑Y坐标
    :param wSrc: 指定图像源位图宽度
    :param hSrc: 指定图像源位图高度
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVSetGrabWindow.argtype = (c_uint64, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int)
    MVGigE.MVSetGrabWindow.restype = c_int
    res = MVGigE.MVSetGrabWindow(c_uint64(hCam), c_int(xDest), c_int(yDest), c_int(wDest), c_int(hDest), c_int(xSrc), c_int(ySrc), c_int(wSrc), c_int(hSrc))
    return res

#MVGetSampleGrab(HANDLE hCam, MVImage* image, int* nFrameID, int msTimeout)
def MVGetSampleGrab(hCam, msTimeout):
    """
     当调用MVFreezeGrabWindow(TRUE)后，调用此函数可以获取当前图像。
    :param hCam: 相机句柄
    :param image: 图像
    :param nFrameID: 图像的ID号
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetSampleGrab.argtype = (c_uint64, c_void_p, c_void_p, c_int)
    MVGigE.MVGetSampleGrab.restype = c_int
    image = c_uint()
    nFrameID = c_int()
    res = MVGigE.MVGetSampleGrab(c_uint64(hCam), byref(image), byref(nFrameID), c_int(msTimeout))
    return res, image.value, nFrameID.value

#MVGetSampleGrabBuf(HANDLE hCam, unsigned char *pImgBuf, unsigned long szBuf,int *pFrameID, int msTimeout)
def MVGetSampleGrabBuf(hCam, pImg, msTimeout):
    """
     当调用MVFreezeGrabWindow(TRUE)后，调用此函数可以获取当前图像。
    :param hCam: 相机句柄
    :param pImgBuf: 图像指针
    :param szBuf: 图像指针内存大小
    :param nFrameID: 图像的ID号
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetSampleGrab.argtype = (c_uint64, POINTER(c_ubyte), c_int, c_void_p, c_int)
    MVGigE.MVGetSampleGrab.restype = c_int
    pImgBuf = cast(pImg.ctypes.data, POINTER(c_ubyte))
    szBuf = pImg.nbytes
    nFrameID = c_int()
    res = MVGigE.MVGetSampleGrabBuf(c_uint64(hCam), pImgBuf, szBuf, byref(nFrameID), c_int(msTimeout))
    return res, nFrameID.value
    
#MVGetDroppedFrame(HANDLE hCam,unsigned long *pDroppedFrames)
def MVGetDroppedFrame(hCam):
    """
     当计算机收到新的图像，而上一帧的Callback函数还没有执行完，SDK中会扔掉新的一帧图像。此函数可以获取扔掉的帧数。
    :param hCam: 相机句柄
    :param pDroppedFrames: 扔掉的帧数
    :return:    MVST_SUCCESS            : 成功
    """
    MVGigE.MVGetDroppedFrame.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetDroppedFrame.restype = c_int
    pDroppedFrames = c_ulong()
    res = MVGigE.MVGetDroppedFrame(c_uint64(hCam), byref(pDroppedFrames))
    return res, pDroppedFrames.value

#MVGetDeviceVendorName(HANDLE hCam,char *pBuf,int *szBuf)
def MVGetDeviceVendorName(hCam):
    """
     获取设备厂商名称
    :param hCam: 相机句柄
    :param pBuf: 用于保存名称的缓冲区，大于等于32字节
    :param [in,out]: szBuf 缓冲区大小
    :return:        MVST_SUCCESS            : 成功    
    """
    MVGigE.MVGetDeviceVendorName.argtype = (c_uint64, c_char_p, c_void_p)
    MVGigE.MVGetDeviceVendorName.restype = c_int
    pBuf = (c_char * 32)()
    szBuf = c_int(32)
    res = MVGigE.MVGetDeviceVendorName(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value

#MVGetDeviceModelName(HANDLE hCam,char *pBuf,int *szBuf)
def MVGetDeviceModelName(hCam):
    """
     获取设备的型号
    :param hCam: 相机句柄
    :param pBuf: 用于保存型号的缓冲区，大于等于32字节
    :param [in,out]: szBuf 缓冲区大小
    :return:    MVST_SUCCESS            : 成功    
    """
    MVGigE.MVGetDeviceModelName.argtype = (c_uint64, c_char_p, c_void_p)
    MVGigE.MVGetDeviceModelName.restype = c_int
    pBuf = (c_char * 32)()
    szBuf = c_int(32)
    res = MVGigE.MVGetDeviceModelName(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value

#MVGetDeviceDeviceID(HANDLE hCam,char *pBuf,int *szBuf)
def MVGetDeviceDeviceID(hCam):
    """
     获取设备的ID号，即序列号
    :param hCam: 相机句柄
    :param pBuf: 用于保存序列号的缓冲区，大于等于16字节
    :param [in,out]: szBuf 缓冲区大小
    :return:    MVST_SUCCESS            : 成功    
    """
    MVGigE.MVGetDeviceDeviceID.argtype = (c_uint64, c_char_p, c_void_p)
    MVGigE.MVGetDeviceDeviceID.restype = c_int
    pBuf = (c_char * 16)()
    szBuf = c_int(16)
    res = MVGigE.MVGetDeviceDeviceID(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value

#MVIsRunning(HANDLE hCam)
def MVIsRunning(hCam):
    """
     相机是否正在采集图像
    :param hCam: 相机句柄
    :return:    正在采集图像返回TRUE,否则返回FALSE
    """
    MVGigE.MVIsRunning.argtype = (c_uint64)
    MVGigE.MVIsRunning.restype = c_bool
    res = MVGigE.MVIsRunning(c_uint64(hCam))
    return res

#MVConvertImage( HANDLE hCam, MVImage* pImageSrc,MVImage* pImageDst )
def MVConvertImage(hCam):
    """
     图像格式转换
    :param hCam: 相机句柄
    :param pImageSrc: 源图像指针
    :param pImageDst: 目标图像指针
    :return:    
    :note:  目前仅支持源图像和目标图像的宽高相同，从16Bit转为8Bit, 从48Bit转为24Bit
    """
    MVGigE.MVConvertImage.argtype = (c_uint64, c_void_p, c_void_p)
    MVGigE.MVConvertImage.restype = c_int
    pImageSrc = c_uint()
    pImageDst = c_uint()
    res = MVGigE.MVConvertImage(c_uint64(hCam), byref(pImageSrc), byref(pImageDst))
    return res, pImageSrc.value, pImageDst.value

#MVCopyImageInfoROI( HANDLE hCam, MV_IMAGE_INFO* pInfoSrc, MV_IMAGE_INFO* pInfoDst, RECT roi )
def MVCopyImageInfoROI(hCam, pInfoSrc, x, y, w, h):
    """
     直接从回调函数传回的图像信息中裁剪出图像的一部分，当相机不支持硬件ROI时，可以用此函数实现软件ROI。
    :param hCam: 相机句柄
    :param pInfoSrc: 源图像指针，一般是回调函数传回的图像信息指针
    :param pInfoDst: 目标图像指针，nPixelType要和源图像的相同，需要提前分配好内存。并且宽度和高度要和roi的宽高相同。
    :param roi: 感兴趣的区域,roi的left,right,top,bottom均须是2的整倍数
    :return:    
    :note:  目前仅支持源图像和目标图像的宽高相同，从16Bit转为8Bit, 从48Bit转为24Bit
    """
    MVGigE.MVCopyImageInfoROI.argtype = (c_uint64, POINTER(MV_IMAGE_INFO), POINTER(MV_IMAGE_INFO), RECT)
    MVGigE.MVCopyImageInfoROI.restype = c_int
    pInfoDst = MV_IMAGE_INFO()
    data = (c_ubyte * w * h)()
    pInfoDst.nSizeX = w
    pInfoDst.nSizeY = h
    pInfoDst.nPixelType = pInfoSrc.nPixelType
    pInfoDst.pImageBuffer = POINTER(c_ubyte)(data[0])
    res = MVGigE.MVCopyImageInfoROI(c_uint64(hCam), pInfoSrc, byref(pInfoDst), RECT(x, y, x+w, y+h))
    return res, pInfoDst

#MVSetSaturation(HANDLE hCam, int nSaturation )
def MVSetSaturation(hCam, nSaturation):
    """
     调节饱和度
    :param hCam: 相机句柄
    :param nSaturation: 饱和度,范围-100到100, -100为黑白，0为原图,100为最鲜艳
    :return:    
    """
    MVGigE.MVSetSaturation.argtype = (c_uint64, c_int)
    MVGigE.MVSetSaturation.restype = c_int
    res = MVGigE.MVSetSaturation(c_uint64(hCam), c_int(nSaturation))
    return res

#MVGetSaturation(HANDLE hCam, int *nSaturation )
def MVGetSaturation(hCam):
    """
        获取当前饱和度
    :param hCam: 相机句柄
    :param nSaturation: 饱和度指针
    :return:  
    """
    MVGigE.MVGetSaturation.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetSaturation.restype = c_int
    nSaturation = c_int()
    res = MVGigE.MVGetSaturation(c_uint64(hCam), byref(nSaturation))
    return res, nSaturation.value

#MVSetColorCorrect(HANDLE hCam, int nColorCorrect )
def MVSetColorCorrect(hCam, nColorCorrect):
    """
     颜色校正
    :param hCam: 相机句柄
    :param nColorCorrect,: 颜色校正模式，目前仅支持0和1,0为不校正，1为校正
    :return:    
    """
    MVGigE.MVSetColorCorrect.argtype = (c_uint64, c_int)
    MVGigE.MVSetColorCorrect.restype = c_int
    res = MVGigE.MVSetColorCorrect(c_uint64(hCam), c_int(nColorCorrect))
    return res

#MVGetColorCorrect(HANDLE hCam, int *nColorCorrect )
def MVGetColorCorrect(hCam):
    """
    　获取当前颜色校正模式
    :param hCam: 相机句柄
    :param nColorCorrect: 颜色校正模式指针
    :return:  
    """
    MVGigE.MVGetColorCorrect.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetColorCorrect.restype = c_int
    nColorCorrect = c_int()
    res = MVGigE.MVGetColorCorrect(c_uint64(hCam), byref(nColorCorrect))
    return res, nColorCorrect.value

#MVRegisterMessage(HANDLE hCam, HWND hWnd, UINT nMsg)
def MVRegisterMessage(hCam, hWnd, nMsg):
    """
     注册用于接收消息的窗口句柄和消息值。当相机断开或重新连上时会发送消息到该窗口。
    :param hCam: 相机句柄
    :param hWnd: 用于接收消息的窗口句柄
    :param nMsg: 消息值
    :return:    
     *  \sa MVEnableMessage
    """
    MVGigE.MVRegisterMessage.argtype = (c_uint64, HWND, UINT)
    MVGigE.MVRegisterMessage.restype = c_int
    res = MVGigE.MVRegisterMessage(c_uint64(hCam), hWnd, nMsg)
    return res

#MVEnableMessage(HANDLE hCam, int nMessageType, bool bEnable)
def MVEnableMessage(hCam, nMessageType, bEnable):
    """
     是否允许发送某个消息
    :param hCam: 相机句柄
    :param nMessageType: 消息类型, MSG_ID_LOST,MSG_ID_RECONNECT
    :param bEnable: 如果为TRUE,则发送该消息，为FALSE则不发送该消息
    :return:    
     *  \sa MVRegisterMessage
    """
    MVGigE.MVEnableMessage.argtype = (c_uint64, c_int, c_bool)
    MVGigE.MVEnableMessage.restype = c_int
    res = MVGigE.MVEnableMessage(c_uint64(hCam), c_int(nMessageType), c_bool(bEnable))
    return res

#MVGetUserDefinedName(HANDLE hCam, char *pBuf,int *szBuf )
def MVGetUserDefinedName(hCam):
    """
    　获取自定义名称
    :param hCam: 相机句柄
    :param pBuf: 相机名称缓冲区
    :param [in/out]: szBuf 相机名称缓冲区长度指针
    :return:  
    """
    MVGigE.MVGetUserDefinedName.argtype = (c_uint64, c_char_p, POINTER(c_int))
    MVGigE.MVGetUserDefinedName.restype = c_int
    pBuf = (c_char * 16)()
    szBuf = c_int(16)
    res = MVGigE.MVGetUserDefinedName(c_uint64(hCam), pBuf, byref(szBuf))
    return res, pBuf.value, szBuf.value

#MVSetUserDefinedName(HANDLE hCam, char *pBuf,int szBuf )
def MVSetUserDefinedName(hCam, pBuf, szBuf):
    """
    　设置自定义名称
    :param hCam: 相机句柄
    :param pBuf: 相机名称缓冲区
    :param szBuf: 相机名称缓冲区长度指针
    :return:  
    """
    MVGigE.MVSetUserDefinedName.argtype = (c_uint64, c_char_p, c_int)
    MVGigE.MVSetUserDefinedName.restype = c_int
    #pBuf = c_char()
    res = MVGigE.MVSetUserDefinedName(c_uint64(hCam), pBuf, c_int(szBuf))
    return res

#MVOpenCamByIndexReadOnly(unsigned char idx,HANDLE *hCam)
def MVOpenCamByIndexReadOnly(idx):
    """
     以只读方式打开相机
    :param idx: idx从0开始，按照相机的IP地址排序，地址小的排在前面。
    :param hCam: 如果成功,返回的相机句柄
    :return:  MVST_INVALID_PARAMETER : idx取值不对
     *          MVST_ACCESS_DENIED      : 相机无法访问，可能正被别的软件控制
     *          MVST_ERROR              : 其他错误
     *          MVST_SUCCESS            : 成功
    """
    MVGigE.MVOpenCamByIndexReadOnly.argtype = (c_ubyte, c_void_p)
    MVGigE.MVOpenCamByIndexReadOnly.restype = c_int
    hCam = c_uint64()
    res = MVGigE.MVOpenCamByIndexReadOnly(c_ubyte(idx), byref(hCam))
    return res, hCam.value

#MVEnumerateAllDevices( int *pDevCnt)
def MVEnumerateAllDevices():
    """
     搜索相机，包含不在同一网段的相机
    :param pDevCnt: 相机数量指针
    :return: 
    """
    MVGigE.MVEnumerateAllDevices.argtype = (c_void_p)
    MVGigE.MVEnumerateAllDevices.restype = c_int
    pDevCnt = c_int()
    res = MVGigE.MVEnumerateAllDevices(byref(pDevCnt))
    return res, pDevCnt.value

#MVForceIp( const char* pMacAddress, const char* pIpAddress, const char* pSubnetMask, const char* pDefaultGateway)
def MVForceIp(pMacAddress, pIpAddress, pSubnetMask, pDefaultGateway):
    """
     为相机设置ip地址
    :param pMacAddress: 待设置ip相机的MAC地址
    :param pIpAddress: 设置给相机的IP地址
    :param pSubnetMask: 设置给相机的子网掩码
    :param pDefaultGateway: 设置给相机的默认网关
    :return: 
    """
    MVGigE.MVForceIp.argtype = (POINTER(c_ubyte), c_char_p, c_char_p, c_char_p)
    MVGigE.MVForceIp.restype = c_int
    res = MVGigE.MVForceIp(pMacAddress, pIpAddress, pSubnetMask, pDefaultGateway)
    return res

#MVGetDevInfo(unsigned char idx,MVCamInfo *pCamInfo)
def MVGetDevInfo(idx):
    """
        获取MVEnumerateAllDevices搜索到的相机的信息
    :param idx: 相机序号
    :param pCamInfo: 相机信息
    :return: 
    """
    MVGigE.MVGetDevInfo.argtype = (c_ubyte, c_void_p)
    MVGigE.MVGetDevInfo.restype = c_int
    pCamInfo = MVCamInfo()
    res = MVGigE.MVGetDevInfo(c_ubyte(idx), byref(pCamInfo))
    return res, pCamInfo

#MVSetPersistentIpAddress( HANDLE hCam, const char* pIpAddress, const char* pSubnetMask, const char* pDefaultGateway)
def MVSetPersistentIpAddress(hCam, pIpAddress, pSubnetMask, pDefaultGateway):
    """
     设置静态IP地址
    :param hCam: 
    :param pIpAddress: 192.168.0.9
    :param pSubnetMask: 255.255.255.0
    :param pDefaultGateway: 0.0.0.0
    :return: 
    """
    MVGigE.MVSetPersistentIpAddress.argtype = (c_uint64, c_void_p, c_void_p, c_void_p)
    MVGigE.MVSetPersistentIpAddress.restype = c_int
    res = MVGigE.MVSetPersistentIpAddress(c_uint64(hCam), pIpAddress, pSubnetMask, pDefaultGateway)
    return res

#MVGetPersistentIpAddress( HANDLE hCam, char* pIpAddress, size_t* pIpAddressLen, char* pSubnetMask, size_t* pSubnetMaskLen, char* pDefaultGateway, size_t* pDefaultGatewayLen)
def MVGetPersistentIpAddress(hCam):
    """
     获取相机的静态IP设置
    :param hCam: 
    :param pIpAddress: 
    :param pIpAddressLen: pIpAddress缓冲区长度
    :param pSubnetMask: 
    :param pSubnetMaskLen: pSubnetMask缓冲区长度
    :param pDefaultGateway: 
    :param pDefaultGatewayLen: pDefaultGateway缓冲区长度
    :return: 
    """
    MVGigE.MVGetPersistentIpAddress.argtype = (c_uint64, c_char_p, c_void_p, c_char_p, c_void_p, c_char_p, c_void_p)
    MVGigE.MVGetPersistentIpAddress.restype = c_int
    pIpAddress = (c_ubyte * 4)()
    pIpAddressLen = c_uint(4)
    pSubnetMask = (c_ubyte * 4)()
    pSubnetMaskLen = c_uint(4)
    pDefaultGateway = (c_ubyte * 4)()
    pDefaultGatewayLen = c_uint(4)
    res = MVGigE.MVGetPersistentIpAddress(c_uint64(hCam), pIpAddress, byref(pIpAddressLen), pSubnetMask, byref(pSubnetMaskLen), pDefaultGateway, byref(pDefaultGatewayLen))
    return res, pIpAddress, pSubnetMask, pDefaultGateway

#MVSetBlackLevelTaps( HANDLE hCam, double fBlackLevel, int nTap )
def MVSetBlackLevelTaps(hCam, fBlackLevel, nTap):
    """
     
    :param hCam: 相机句柄
    :param fBlackLevel: 偏置
    :param nTap: 通道
    :return:    
    """
    MVGigE.MVSetBlackLevelTaps.argtype = (c_uint64, c_double, c_int)
    MVGigE.MVSetBlackLevelTaps.restype = c_int
    res = MVGigE.MVSetBlackLevelTaps(c_uint64(hCam), c_double(fBlackLevel), c_int(nTap))
    return res

#MVGetBlackLevelTaps( HANDLE hCam, double *pBlackLevel,int nTap )
def MVGetBlackLevelTaps(hCam, nTap):
    """
     
    :param hCam: 相机句柄
    :param pBlackLevel: 偏置
    :param nTap: 通道
    :return:    
    """
    MVGigE.MVGetBlackLevelTaps.argtype = (c_uint64, c_void_p, c_int)
    MVGigE.MVGetBlackLevelTaps.restype = c_int
    pBlackLevel = c_double()
    res = MVGigE.MVGetBlackLevelTaps(c_uint64(hCam), byref(pBlackLevel), c_int(nTap))
    return res, pBlackLevel.value

#MVSetBlackLevel( HANDLE hCam, double fBlackLevel)
def MVSetBlackLevel(hCam, fBlackLevel):
    """
     设置偏置
    :param hCam: 相机句柄
    :param fBlackLevel: 偏置
    :return:    
    """
    MVGigE.MVSetBlackLevel.argtype = (c_uint64, c_double)
    MVGigE.MVSetBlackLevel.restype = c_int
    res = MVGigE.MVSetBlackLevel(c_uint64(hCam), c_double(fBlackLevel))
    return res

#MVGetBlackLevel( HANDLE hCam, double *pBlackLevel)
def MVGetBlackLevel(hCam):
    """
     读取偏置
    :param hCam: 相机句柄
    :param pBlackLevel: 偏置
    :return:    
    """
    MVGigE.MVGetBlackLevel.argtype = (c_uint64, c_void_p)
    MVGigE.MVGetBlackLevel.restype = c_int
    pBlackLevel = c_double()
    res = MVGigE.MVGetBlackLevel(c_uint64(hCam), byref(pBlackLevel))
    return res, pBlackLevel.value

