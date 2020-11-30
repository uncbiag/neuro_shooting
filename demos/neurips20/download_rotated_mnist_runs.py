import gdown
import tarfile

url = 'https://drive.google.com/uc?id=1jyPcw-ouwxdbHpgubUB5vTxn8DUviD7k'
output = 'rotated_mnist_runs.tar.gz'
gdown.download(url, output, quiet=False)

tar = tarfile.open('rotated_mnist_runs.tar.gz', "r:gz")
tar.extractall()
tar.close()