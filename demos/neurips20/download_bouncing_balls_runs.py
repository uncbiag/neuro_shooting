import gdown
import tarfile

url = 'https://drive.google.com/uc?id=1Unw9AtbgKtoN7pzKZxZK-M4CMklvPlWd'
output = 'bouncing_balls_runs.tar.gz'
gdown.download(url, output, quiet=False)

tar = tarfile.open('bouncing_balls_runs.tar.gz', "r:gz")
tar.extractall()
tar.close()
