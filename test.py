import settings
import data
import pandas as pd

catchphrase_dataloader = data.make_dataloader(settings.CATCHPHRASE_CSV, "catchphrase")
memeimage_dataloader = data.make_dataloader(settings.MEMEIMAGE_CSV, "memeimage")


batch = next(iter(memeimage_dataloader))

import pdb; pdb.set_trace()
print(batch)

dataset = pd.read_csv(settings.CATCHPHRASE_CSV)