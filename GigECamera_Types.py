from enum import Enum
from ctypes import *

# 
INT64_MAX = 0x7fffffffffffffff  #/*maximum signed __int64 value */
INT64_MIN = 0x8000000000000000  #/*minimum signed __int64 value */
UINT64_MAX = 0xffffffffffffffff  #/*maximum unsigned __int64 value */

INT32_MAX = 0x000000007fffffff  #/*maximum signed __int32 value */
INT32_MIN = 0xffffffff80000000  #/*minimum signed __int32 value */
UINT32_MAX = 0x00000000ffffffff  #/*maximum unsigned __int32 value */

INT8_MAX = 0x000000000000007f  #/*maximum signed __int8 value */
INT8_MIN = 0xffffffffffffff80  #/*minimum signed __int8 value */
UINT8_MAX = 0x00000000000000ff  #/*maximum unsigned __int8 value */

FALSE = 0
TRUE = 1

# Bayer��ɫģʽ
# MV_BAYER_MODE
BayerRG = 0	#< ��ɫģʽRGGB
BayerBG = 1	#< ��ɫģʽBGGR
BayerGR = 2	#< ��ɫģʽGRBG
BayerGB = 3	#< ��ɫģʽGBRG
BayerGRW = 4	#< ��ɫģʽGRW
BayerInvalid = 5
 
# ͼ�����ظ�ʽ
# MV_PixelFormatEnums
PixelFormat_Mono8 = 0x01080001	#!<8Bit�Ҷ�
PixelFormat_BayerBG8=0x0108000B	#!<8Bit Bayerͼ,��ɫģʽΪBGGR
PixelFormat_BayerRG8=0x01080009	#!<8Bit Bayerͼ,��ɫģʽΪRGGB
PixelFormat_BayerGB8=0x0108000A	#!<8Bit Bayerͼ,��ɫģʽΪGBRG
PixelFormat_BayerGR8=0x01080008	#!<8Bit Bayerͼ,��ɫģʽΪGRBG
PixelFormat_BayerGRW8=0x0108000C	#!<8Bit Bayerͼ,��ɫģʽΪGRW8
PixelFormat_Mono16=0x01100007	#!<16Bit�Ҷ�
PixelFormat_BayerGR16=0x0110002E	#!<16Bit Bayerͼ,��ɫģʽΪGR
PixelFormat_BayerRG16=0x0110002F	#!<16Bit Bayerͼ,��ɫģʽΪRG
PixelFormat_BayerGB16=0x01100030	#!<16Bit Bayerͼ,��ɫģʽΪGB
PixelFormat_BayerBG16=0x01100031	#!<16Bit Bayerͼ,��ɫģʽΪBG

 
# ���󷵻�ֵ����
# MVSTATUS_CODES 
MVST_SUCCESS                = 0      #/< û�д���      
MVST_ERROR                  = -1001  #/< һ�����
MVST_ERR_NOT_INITIALIZED    = -1002  #!< û�г�ʼ��
MVST_ERR_NOT_IMPLEMENTED    = -1003  #!< û��ʵ��
MVST_ERR_RESOURCE_IN_USE    = -1004  #!< ��Դ��ռ��
MVST_ACCESS_DENIED          = -1005  #/< �޷�����
MVST_INVALID_HANDLE         = -1006  #/< ������
MVST_INVALID_ID             = -1007  #/< ����ID
MVST_NO_DATA                = -1008  #/< û������
MVST_INVALID_PARAMETER      = -1009  #/< �������
MVST_FILE_IO                = -1010  #/< IO����
MVST_TIMEOUT                = -1011  #/< ��ʱ
MVST_ERR_ABORT              = -1012  #/< �˳�
MVST_INVALID_BUFFER_SIZE    = -1013  #/< �������ߴ����
MVST_ERR_NOT_AVAILABLE      = -1014  #/< �޷�����
MVST_INVALID_ADDRESS        = -1015  #/< ��ַ����

# TriggerSourceEnums
TriggerSource_Software=0#!<����ģʽ�£���������(����ָ��)�������ɼ�
TriggerSource_Line1=2 #!<����ģʽ�£����ⴥ���ź��������ɼ�

# TriggerModeEnums
TriggerMode_Off = 0  #!<����ģʽ�أ���FreeRunģʽ����������ɼ�
TriggerMode_On = 1	#!<����ģʽ��������ȴ����������ⴥ���ź��ٲɼ�ͼ��

# TriggerActivationEnums 
TriggerActivation_RisingEdge = 0  #!<�����ش���
TriggerActivation_FallingEdge = 1 #!<�½��ش���

# LineSourceEnums
LineSource_Off=0  #!<�ر�
LineSource_ExposureActive=5  #!<���ع�ͬʱ
LineSource_Timer1Active=6	#!<�ɶ�ʱ������
LineSource_UserOutput0=12	#!<ֱ������������

# UserSetSelectorEnums
UserSetSelector_Default = 0  #!<��������
UserSetSelector_UserSet1 = 1  #!<�û�����1
UserSetSelector_UserSet2 = 2   #!<�û�����2

# SensorTapsEnums
SensorTaps_One = 0  #!<��ͨ��
SensorTaps_Two = 1  #!<˫ͨ��
SensorTaps_Three = 2  #!<��ͨ��
SensorTaps_Four = 3  #!<��ͨ��

# AutoFunctionProfileEnums
AutoFunctionProfile_GainMinimum = 0  #!<Keep gain at minimum
AutoFunctionProfile_ExposureMinimum = 1  #!<Exposure Time at minimum


# GainAutoEnums
GainAuto_Off = 0  #!<Disables the Gain Auto function.
GainAuto_Once = 1  #!<Sets operation mode to 'once'.
GainAuto_Continuous = 2   #!<Sets operation mode to 'continuous'.


#! ExposureAutoEnums
ExposureAuto_Off = 0 #!<Disables the Exposure Auto function.
ExposureAuto_Once = 1  #!<Sets operation mode to 'once'.
ExposureAuto_Continuous = 2   #!<Sets operation mode to 'continuous'.


# BalanceWhiteAutoEnums
BalanceWhiteAuto_Off = 0  #!<Disables the Balance White Auto function.
BalanceWhiteAuto_Once = 1  #!<Sets operation mode to 'once'.
BalanceWhiteAuto_Continuous = 2   #!<Sets operation mode to 'continuous'.


#####################################
# ImageFlipType
FlipHorizontal = 0  #!< ���ҷ�ת
FlipVertical = 1 #!< ���·�ת
FlipBoth = 2 #!< ��ת180��


# ImageRotateType
Rotate90DegCw = 0       #!< ˳ʱ����ת90��
Rotate90DegCcw = 1       #!< ��ʱ����ת90��


# TransferControlModeEnums
TransferControlMode_Basic = 0,
TransferControlMode_UserControlled = 2

#! EventIdEnums
EVID_LOST = 0,	#!< �¼�ID������Ͽ�
EVID_RECONNECT = 1	#! �¼�ID���������������


###################################################################

class RECT(Structure):
	_fields_ = [('left', c_long),          
                ('top', c_long),  
                ('right', c_long),          
                ('bottom', c_long)]
				
#�ص������õ������ݵĽṹ
class MV_IMAGE_INFO(Structure):
    _fields_ = [('nTimeStamp', c_ulonglong),         # ʱ������ɼ���ͼ���ʱ�̣�����Ϊ0.01us
                ('nBlockId', c_ushort),              # ֡�ţ��ӿ�ʼ�ɼ���ʼ����
                ('pImageBuffer', POINTER(c_ubyte)),  # ͼ��ָ�룬��ָ��(0,0)���������ڴ�λ�õ�ָ�룬ͨ����ָ����Է�������ͼ
                ('nImageSizeAcq', c_ulong),          # �ɼ�����ͼ���С[�ֽ�]
                ('nMissingPackets', c_ubyte),        # ��������ж����İ�����
                ('nPixelType', c_ulonglong),         # ͼ���ʽ
                ('nSizeX', c_uint),                  # ͼ�����
                ('nSizeY', c_uint),                  # ͼ��߶�
                ('nOffsetX', c_uint),                # ͼ��ˮƽ����
                ('nOffsetY', c_uint)]                # ͼ��ֱ����

# �������Ϣ
class MVCamInfo(Structure):
    _fields_ = [('mIpAddr', c_ubyte*4),           # �����IP��ַ
                ('mEthernetAddr', c_ubyte*6),     # �����MAC��ַ
                ('mMfgName', c_char*32),          # ����ĳ�������
                ('mModelName', c_char*32),        # ����ͺ�
                ('mSerialNumber', c_char*16),     # ������к�
				('mUserDefinedName', c_char*16),  # �û������������
                ('m_IfIP', c_ubyte*4),            # ������������������IP��ַ
                ('m_IfMAC', c_ubyte*6)]           # �������������ӵ�����MAC��ַ

			
class MVStreamStatistic(Structure):
    _fields_ = [('m_nTotalBuf', c_ulong),         # �ӿ�ʼ�ɼ����ܼƳɹ��յ������ͼ��֡��
                ('m_nFailedBuf', c_ulong),        # �ӿ�ʼ�ɼ����ܼ��յ��Ĳ����ͼ��֡��
                ('m_nTotalPacket', c_ulong),      # �ӿ�ʼ�ɼ����ܼ��յ���ͼ�����ݰ���
                ('m_nFailedPacket', c_ulong),     # �ӿ�ʼ�ɼ����ܼƶ�ʧ��ͼ�����
                ('m_nResendPacketReq', c_ulong),  # �ӿ�ʼ�ɼ����ܼ��ط������ͼ�����ݰ���
                ('m_nResendPacket', c_ulong)]     # �ӿ�ʼ�ɼ����ܼ��ط��ɹ���ͼ�����ݰ���

MVStreamCB = WINFUNCTYPE(c_int, POINTER(MV_IMAGE_INFO), POINTER(c_longlong))
	