import requests
import pandas as pd
import string
import re

def get(DEVELOPER_KEY = "API_KEY",VIDEO_ID = "hT_nvWreIhg",size=100):
    url=f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults={size}&videoId={VIDEO_ID}&key={DEVELOPER_KEY}"
    data = requests.get(url)
    if data.status_code==200:
        return data.json(),VIDEO_ID

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
def comments(data,vid):
    comments=[]
    for key,row in data.items():
        if isinstance(row,list):
            for item in row:
                for d,v in item.get('snippet').items():
                    if isinstance(v,dict):
                        for k,v2 in v.items():
                            if isinstance(v2,dict):
                                for k2,v3 in v2.items():
                                    if k2=='textOriginal':
                                        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', v3, flags=re.MULTILINE)
                                        text= text.translate(str.maketrans('', '', string.punctuation))
                                        comments.append({
                                            'comment':deEmojify(text),
                                            'vid':vid,
                                        })
    return comments

if __name__ == "__main__":
    data,vid =get()
    
    commentsList = comments(data,vid)
    print(commentsList)