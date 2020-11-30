import gdown
import tarfile

url = 'https://drive.google.com/uc?id=1HmH2_ZH-7bWO-S7ThsxuZwzxcj2CWcf8'
output = 'bouncing_balls.tar.gz'
gdown.download(url, output, quiet=False)

tar = tarfile.open('bouncing_balls.tar.gz', "r:gz")
tar.extractall()
tar.close()
