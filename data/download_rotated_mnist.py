import gdown
import tarfile

url = 'https://drive.google.com/uc?id=1FIX1ZXtbYSM3cmGqrM95INkKEkramMFK'
output = 'rot_mnist.tar.gz'
gdown.download(url, output, quiet=False)

tar = tarfile.open('rot_mnist.tar.gz', "r:gz")
tar.extractall()
tar.close()
