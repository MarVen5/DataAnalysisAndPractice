# **数据分析及实践 Lab 2**

马文宇 PB23061139

### **PART 1**

进入 `Nature` 主页，通过高级检索功能，搜索关键词 `llm`，限制年份为 `2023-2024`，搜索得到近两年间关键词含有 `llm` 的文章，使用 `python` 代码检索结果第一页的 `html` 内容并解码为文本串，将其以 `UTF-8` 编码格式写入 `page1.txt` 文件中，用于后续处理。

1. **构造搜索 URL**：通过分析 *Nature* 搜索页面的 URL 结构，构造带有 `q=llm`（关键词）、`date_range=2023-2024`（时间范围）和 `page=1`（第一页）的 URL。  
2. **发送 HTTP 请求**：使用 `requests` 库向 *Nature* 网站发送 GET 请求，获取返回的 HTML 页面内容。  
3. **设置编码**：确保网页内容以 `UTF-8` 编码正确解析，避免乱码问题。  
4. **保存网页内容**：将 HTML 源码保存至本地 `page1.txt` 文件，为后续处理（如解析文章链接、提取作者信息等）做准备。  

```python
import requests # 用于发送 HTTP 请求
from bs4 import BeautifulSoup # 用于解析 HTML 内容

url = 'https://www.nature.com/search?q=llm&order=relevance&date_range=2023-2024&page=1'

response = requests.get(url) # 发送 GET 请求获取网页内容
response.encoding = 'utf-8' # 设置响应的编码为 UTF-8

with open ('page1.txt', 'w', encoding = 'utf-8') as f: # 打开文件 'page1.txt'，以写入模式保存页面内容
    f.write(response.text)  # 将网页的 HTML 内容写入文件
```

### **PART 2**

打开 `page1.txt`，观察相关数据的组织格式规律。从这些文本串中提取一些论文相关的重要信息（包括文章标题，文章地址，文章简介，作者列表，文章类型，期刊名称，卷宗/页面信息），并按照发表的期刊名进行分类，存储到一个字典列表中。基于以上结果，输出每个期刊在 `page1` 中包含的论文数量。

1. **解析 HTML 文件**：  
   - 读取 `page1.txt` 内容，并使用 `BeautifulSoup` 解析 HTML 结构。  
2. **提取论文信息**：  
   - 使用 `find_all('article')` 遍历所有文章节点，逐步提取各项关键信息。  
   - **标题、简介** 通过 `h3` 和 `p` 标签提取。  
   - **链接** 从 `a` 标签的 `href` 属性获取。  
   - **作者信息** 通过 `ul` 标签获取多个 `li` 元素。  
   - **文章类型、期刊名称、卷宗/页面信息** 通过 `c-meta` 相关类名提取。  
3. **按期刊名称分类**：  
   - 以期刊名称作为键，存储对应的论文列表。  
   - 使用 `setdefault()` 确保字典结构统一。  
4. **数据存储与统计**：  
   - 论文信息以 `JSON` 格式存储到 `nature_llm_before.json` 文件中。  
   - 统计并输出每个期刊包含的论文数量。

```python
from bs4 import BeautifulSoup
import json

# 解析 HTML 内容
soup = BeautifulSoup(response.text, 'html.parser')

# 存储期刊论文信息的字典
journal_paper_dict = {}

# 遍历所有文章
for article in soup.find_all('article'):
    # 定义一个通用函数，用于获取指定标签的文本内容
    def get_element(cls=None, tag=None, default=""):
        """获取指定HTML标签的文本内容，若不存在则返回默认值"""
        element = article.find(tag, class_=cls) if cls else article.find(tag)
        return element.text.strip() if element else default

    # 提取文章标题，如果不存在则返回 "No Title"
    title = get_element(tag='h3', default="No Title")

    # 提取文章简介，如果不存在则返回 "no description"
    description = get_element(tag='p', default="no description")

    # 提取文章链接，如果没有链接，则返回 "#"
    url = (article.find('a')['href'] if article.find('a') else "#").strip()

    # 提取文章作者信息
    authors_ul = article.find('ul', class_="c-author-list c-author-list--compact c-author-list--truncated")
    authors_compact = [li.text.strip() for li in authors_ul.find_all('li')] if authors_ul else []

    # 查找包含元数据信息的 div
    meta_items = article.find('div', class_="c-card__section c-meta") or {}

    # 提取文章类型、期刊名称、卷宗/页面信息
    meta_data = {
        'type': meta_items.find('span', class_="c-meta__type"),  # 文章类型
        'journal': meta_items.find('div', class_="c-meta__item--block-at-lg"),  # 期刊名称
        'volume': meta_items.find('div', class_="c-meta__item--block-at-lg")  # 卷宗/页面信息
    }

    # 组织每篇文章的信息
    paper_info = {
        "title": title,
        "authors": authors_compact or ["Anonymous"],  # 若无作者信息，则标记为 "Anonymous"
        "url": f"{url}",
        "description": description,
        "type": meta_data['type'].text.strip() if meta_data['type'] else "Unknown",
        "volume_page_info": meta_data['volume'].text.strip() if meta_data['volume'] else ""
    }

    # 获取期刊名称，若无信息则标记为 "Unknown Journal"
    journal = meta_data['journal'].text.strip() if meta_data['journal'] else "Unknown Journal"

    # 使用 setdefault 避免重复判断 journal 是否在字典中
    journal_paper_dict.setdefault(journal, {"journal": journal, "papers": []})["papers"].append(paper_info)

# 将字典转换为列表格式
article_list = list(journal_paper_dict.values())

# 将数据保存为 JSON 文件
json.dump(article_list, open('nature_llm_before.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

# 按格式输出每个期刊的论文数量
for journal in article_list:
    print(f'{journal["journal"]:<50}The number of papers: {len(journal["papers"])}')

```

文章的结构已经存储在 `nature_llm_before.json` 中，以及预期的输出如下：

```
Nature Machine Intelligence                       The number of papers: 6
Nature Communications                             The number of papers: 10
Scientific Reports                                The number of papers: 12
Nature Methods                                    The number of papers: 1
npj Digital Medicine                              The number of papers: 3
Nature Medicine                                   The number of papers: 2
Nature                                            The number of papers: 4
Nature Human Behaviour                            The number of papers: 2
npj Precision Oncology                            The number of papers: 1
Nature Computational Science                      The number of papers: 2
BDJ Open                                          The number of papers: 1
Nature Reviews Urology                            The number of papers: 1
Communications Materials                          The number of papers: 1
npj Biodiversity                                  The number of papers: 1
Humanities and Social Sciences Communications     The number of papers: 1
Eye                                               The number of papers: 1
npj Computational Materials                       The number of papers: 1
```

### **PART 3**

观察可以发现该文章的作者信息实际上多于搜索结果中展示的内容，请你仔细观察此界面的 `html` 数据组织格式，依据此编写 `python` 程序，将上一步骤中的字典提取内容中的作者列表中的内容进行替换，替换为文章主页面显示的全部作者。

在 **PART 2** 中，我们从 `page1.txt` 中提取了论文的部分信息，包括标题、简介、链接、作者等。然而，搜索结果页面展示的作者信息并不完整，需要进一步访问每篇文章的主页面，获取 **全部作者**。因此，执行以下步骤：
1. **获取主页面 HTML**：对每篇文章的链接发送请求，获取文章的主页面 HTML 内容。
2. **解析 JSON-LD 数据**：通过解析页面中的 `application/ld+json` 数据（通常包含文章的详细元信息），从中提取文章的所有作者信息。
3. **更新作者信息**：将每篇文章中部分作者信息替换为完整的作者列表，更新字典结构。
4. **存储更新后的数据**：将更新后的字典列表转化为 JSON 格式，并保存至文件 `nature_llm.json`。

```python
import json
import requests
from bs4 import BeautifulSoup

def get_author_from_jsonld(json_data):
    return [author["name"] for author in json_data["mainEntity"]["author"]]

def fetch_article_details(url):
    full_url = f'https://www.nature.com{url}' if not url.startswith('http') else url
    response = requests.get(full_url, timeout=10)
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response.text

def parse_jsonld(html):
    soup = BeautifulSoup(html, 'html.parser')
    script = soup.find('script', type="application/ld+json")
    if script:
        try:
            return json.loads(script.string.strip())
        except json.JSONDecodeError:
            pass
    return None

for journal_paper in article_list:
    for paper in journal_paper["papers"]:
        html = fetch_article_details(paper["url"])
        if not html:
            continue
            
        json_data = parse_jsonld(html)
        if json_data:
            paper["authors"] = get_author_from_jsonld(json_data)
        else:
            paper["authors"] = ["Authors not found"]

json.dump(article_list, open('nature_llm.json.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
```

### **PART 4**

将上一步获得的字典列表转化为 `json` 对象，并以 2 字符缩进的方式写入 `nature llm.json` 文件中。已经在 **PART 3** 代码中实现。