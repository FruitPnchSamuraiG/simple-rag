import re

def find_urls(text):
    url_pattern = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
        r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+'
        r'(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    )

    urls = re.findall(url_pattern, text)
    
    urls = [url[0] for url in urls]
    
    return urls


text = """
help me find out whihc countries counselling is provided here https://www.inspiruseducation.com/resources/post-graduate-counselling-resources/
"""

found_urls = find_urls(text)
print("Found URLs:", found_urls)
