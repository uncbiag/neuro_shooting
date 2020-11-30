import gdown
import tarfile


url = 'https://drive.google.com/uc?id=1zzRGsKiUD6nPjJ48eomcXqYn0tBNA6DX'
output = 'concentric_circles_runs.tar.gz'
gdown.download(url, output, quiet=False)

tar = tarfile.open('concentric_circles_runs.tar.gz', "r:gz")
tar.extractall()
tar.close()
