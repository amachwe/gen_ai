import xml.etree.ElementTree as et
from lxml import etree
from bs4 import BeautifulSoup

IN_FILE = "data/post_data.xml"
OUT_FILE = "./data/posts/blog_text"

parser = etree.XMLParser(remove_comments=True)
tree = etree.parse(IN_FILE,parser=parser)

skip_count = 0

file_count = 0
files_written = []
for item in tree.getiterator():
    if item.tag == '{http://purl.org/rss/1.0/modules/content/}encoded':
        lines = []
        file_count+=1
        for line in item.text.splitlines():
            soup = BeautifulSoup(line)
            line = soup.get_text()
            if line[:3] == "<!-" and line[-2:]=="->":
                continue
            lines.append(line)
        if len(lines) > 5:
            file_name = f"{OUT_FILE}_{file_count}.txt"
            with open(file_name, 'w', encoding='utf-8') as fh:
                try:
                    fh.writelines(lines)
                    files_written.append(file_name)
                except Exception as e:
                    print(e)
                    skip_count += 1
                    print("Skipping:", line)
                    print("----------------------------\n")


with open("./data/posts/blog_post_index.txt","w") as fh:
    for line in files_written:
        fh.write(line)
        fh.write(",")

                    


print("Done.", "\tLines skipped:", skip_count)






                
            
            
        




