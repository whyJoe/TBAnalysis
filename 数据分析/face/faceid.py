from aip import AipFace
import base64

APP_ID = '16805658'
API_KEY = '2bT2pIdLA7RYDWBhN1qxtGNn'
SECRET_KEY = 'mWNjsKTDn0kGeyKRyRxWbrTxtAZyKA0X'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)


def openfile(filepath):
    with open(filepath, 'rb') as f:
        image = base64.b64encode(f.read())
        result = str(image, 'utf-8')
    return result


# 选择以BASE64传输
image_type = "BASE64"

# 可选项
options = {"face_field": "beauty"}

# 此处的返回值为人脸的基本检测的数值效果
print(client.detect(openfile("lzh.jpg"), image_type, options))

# 人脸检测
# 在用户组中检索
groupIdList = "faceid1"
#
# """ 调用人脸搜索 """

# result = client.search(openfile("9b9a2f6e2659983f55ef598b6e7b5be.jpg"), image_type, groupIdList)
# print(result['result'])


# 人脸对比
# def face_match(filepath1, filepath2):
#     result1 = client.match([
#         {
#             'image': str(base64.b64encode(open(filepath1, 'rb').read()), 'utf-8'),
#             'image_type': 'BASE64',
#         },
#         {
#             'image': str(base64.b64encode(open(filepath2, 'rb').read()), encoding='utf-8'),
#             'image_type': 'BASE64',
#         }
#     ])
#     print(result1)
#     a = result1['result']['score']
#     print(a)
#
#
# face_match('self.jpg', '9b9a2f6e2659983f55ef598b6e7b5be.jpg')
