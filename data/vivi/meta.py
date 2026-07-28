import json
from glob import glob

fout_train = open('train.list', 'w')
fout_dev = open('dev.list', 'w')

videos = {}
for video in glob('/nfs-speech-cfs/wangzhou/data/tts/VividHead/videos/*'):
    key = video.split('/')[-1].split('.')[0]
    videos[key] = video

audios = {}
for audio in glob('/nfs-speech-cfs/wangzhou/data/tts/VividHead/audios/*'):
    key = audio.split('/')[-1].split('.')[0]
    
    audios[key] = audio
print(len(videos), len(audios))

data = []
for key in audios.keys():
    audio = audios[key]
    video = videos[key]
    res = dict(key = key, audio=audio, video=video)
    res = json.dumps(res, ensure_ascii=False)
    data.append(res)

for item in data[:-100]:
    fout_train.write("%s\n" % item)
fout_train.close()

for item in data[-100:]:
    fout_dev.write("%s\n" % item)
fout_dev.close()