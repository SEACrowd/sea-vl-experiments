import glob
from photos_where import Where

# Extract EXIF
where = Where(
    pictures_root=glob.glob('*'),
    feather_location='photos.feather',
    processes=32
)

# Data Filtering
df = where.raw_exif_df
raw_df = df[df['filename'].str.contains('.jpg')] # Filtering for Madaa.ai dataset as the original images are all in `.jpg` format

# Save Missing Info
raw_df.isna().sum().to_csv('missing_info_raw.csv')
