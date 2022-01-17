from PIL import ImageCms
from pathlib import Path

path = (Path(__file__).parent / 'ProPhoto.icm')
ICC_PROPHOTO_RGB_PROFILE_BYTES = ImageCms.getOpenProfile(str(path)).tobytes()
