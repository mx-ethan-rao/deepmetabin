
gmvae_path = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/deepmetabin/gmvae_1000/gmm_epoch_600.csv'
other_path = '/tmp/local/zmzhang/DeepMetaBin/CAMI1/low/metadecoder/nc_result.csv'
proces_num = 21

gmvae_dict = {}
with open(gmvae_path, 'r') as f:
    for l in f.readlines():
        items = l.split()
        temp = items[1].split('_')
        gmvae_dict[temp[0].strip() + '_' + temp[1].strip()] = items[0].strip()

classs = set()
for val in gmvae_dict.values():
    classs.union(set(val))
num__class = len(classs) + 1

other_dict = {}
with open(other_path, 'r') as f:
    for l in f.readlines():
        items = l.split(',')
        gmvae_dict[items[0].strip()] = items[1].strip()


