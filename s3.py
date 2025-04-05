import boto3
from botocore.client import Config
from urllib.parse import urlparse, urlunparse
from config import config

class S3ClientWrapper:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config.s3_endpoint,
            aws_access_key_id=config.s3_ak,
            aws_secret_access_key=config.s3_sk,
            config=Config(s3={'addressing_style': 'virtual'})
        )
        self.bucket = config.s3_bucket

    def list_files_with_prefix(self, prefix=''):
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix
        )

        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []

    def generate_presigned_url(self, key: str, expires_in: int = 1800) -> str:
        url = self.s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': self.bucket,
                'Key': key
            },
            ExpiresIn=expires_in
        )

        return url

if __name__ == '__main__':
    s3 = S3ClientWrapper()

    # 获取某个前缀下的文件
    keys = s3.list_files_with_prefix(prefix="processed_1080_png/charts/",)

    # 生成签名链接
    for key in keys:
        url = s3.generate_presigned_url(key)
        print(f"{key} -> {url}")
