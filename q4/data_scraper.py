from bs4 import BeautifulSoup
import requests

url="https://www.stat.cmu.edu/~larry/all-of-statistics/=data/glass.dat"
response = requests.get(url, verify=False)

soup=BeautifulSoup(response.text, "html.parser")    
with open("data.csv", "w") as file:
    file.write(soup.prettify())

with open("data.csv", "r") as file:
    data=file.readlines()
    print(len(data))
    data=[line.split() for line in data] 
    data=[",".join(line) for line in data]

with open("data.csv", "w") as file:
    for line in data:
        file.write(line)
        file.write("\n")