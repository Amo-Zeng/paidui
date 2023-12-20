import sys
import json
import os
sys.path.insert(0, "./BGE")

from tool import SearchTool

BGE_DATA_PATH = "./BGE/data/ai_filter.json"
BGE_ABSTRACT_EMB_PATH = "./BGE/data/abstract.npy"
BGE_ABSTRACT_INDEX_PATH = "./BGE/abstract.index"
BGE_ABSTRACT_BM25_INDEX_PATH = "./BGE/abstract_bm25_index"
BGE_META_EMB_PATH = "./BGE/data/meta.npy"
BGE_META_INDEX_PATH = "./BGE/meta.index"
BGE_META_BM25_INDEX_PATH = "./BGE/meta_bm25_index"
BGE_BATCH_SIZE = 128
BGE_SEARCH_NUM = 3


# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """The following information is known:
{context}

Based on the above known information, answer the user's questions concisely and professionally. If you can't get an answer from it, please say "the question cannot be answered based on known information" or "not enough relevant information is provided". Adding fabricated components to the answer is not allowed, and the answer should be in English. The question is：{question}"""
def generate_prompt(related_docs, query, prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = []
    for idx, doc in related_docs.items():
        content = "The abstract is: " + doc["content"]["abstract"].replace("\n", " ") + " " + "The title is: " + doc["content"]["title"].replace(
            "\n", " ") + " " + "The author is: " + ",".join(doc["content"]["authors"]) + "."
        content = "source "+str(idx) + ": " + content

        context.append(content)
    context = "\n\n".join(context)

    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt



def get_search_result_from_BGE(query):
    #print(query)
    search_tool = SearchTool(BGE_DATA_PATH,
                         BGE_ABSTRACT_EMB_PATH,
                         BGE_ABSTRACT_INDEX_PATH,
                         BGE_ABSTRACT_BM25_INDEX_PATH,
                         BGE_META_EMB_PATH,
                         BGE_META_INDEX_PATH,
                         BGE_META_BM25_INDEX_PATH,
                         128)

    input_text = query
    retrieval_type = "merge"
    query_type = "by query"
    target_type = "conditional"
    num = BGE_SEARCH_NUM
    rerank = "enable"
    rerank_num = 25
    response = search_tool.search(input_text, retrieval_type, query_type, target_type, num, rerank, rerank_num)
    if len(response) > 0:
        prompt = generate_prompt(response, query)
    else:
        prompt = query
    return prompt

    


#res=get_search_result_from_BGE("find paper about transformers")
#print(res)
def preprossbge(file):
    file=os.path.abspath(file)
    os.system("bash ./BGE/preprocess.sh "+file)
#preprossbge("./BGE/data_demo.json")
函子={"arxiv问答":get_search_result_from_BGE,"百科搜索":get_search_result_from_BGE,"生成数据库":preprossbge}#接受一个变量的函数字典
函丑={}#接受两个变量的函数字典，依此类推
函寅={}
函卯={}
函辰={}
函巳={}
函午={}
函未={}
函申={}
函酉={}
函戌={}
函亥={}
函括={}#接受任意个变量的函数字典，引用函数时需要用括号把函数和变量括起来。
