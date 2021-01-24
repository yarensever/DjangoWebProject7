"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest

import pandas as pd
import requests
import json
#first of all, create one year data

def EPIAS_API():
    down = './test.json'
    url  = 'https://seffaflik.epias.com.tr/transparency/service/market/day-ahead-mcp?endDate=2021-01-24&startDate=2020-01-26'
    outpath=down
    generatedURL=url
    response = requests.get(generatedURL)
    if response.status_code == 200:
        with open(outpath, "wb") as out:
            for chunk in response.iter_content(chunk_size=128):
                out.write(chunk)
    with open(down) as json_file:
        data = json.load(json_file)
    body=data.get('body')
    gen=body.get('dayAheadMCPList')
    df=pd.DataFrame(gen)
    return(df)



def home(request):
    df = EPIAS_API()
    mcp_eur = df['priceEur'].values.tolist()
    mcp_usd = df['priceUsd'].values.tolist()
    mcp_tl  = df['price'].values.tolist()
    print(len(df))
    return render(
        request,
        'app/index.html',
        {
            'mcp_eur':mcp_eur,
            'mcp_usd':mcp_usd,
            'mcp_tl': mcp_tl
        }
    )

