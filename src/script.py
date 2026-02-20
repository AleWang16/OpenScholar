from src.open_scholar import *

from src.use_search_apis import *

# oc = OpenScholar(model="OpenScholar/Llama-3.1_OpenScholar-8B")
# item = {"input": "what are the latest developments in biomed"}
# print(oc.run(item=item, ranking_ce=True))
link = ""
paragraphs = parsing_paragraph(link)

print(paragraphs)

