#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import torch
import os
import sys
import readline
import threading
from threading import Thread
from transformers import TextIteratorStreamer

#model_name = '/media/amo/e/.cache/models--mosaicml--mpt-7b-8k/snapshots/539960c3beb28496cee315cbe1fae3b6aa655837/'
model_name = '/media/amo/e/chat/openchat_3.5/'
resume_from_checkpoint='./微调阿福/'

#device = 'cuda'
device = 'cpu'
max_new_tokens = 1000    # 每轮对话最多生成多少个token
history_max_len = 8000  # 模型记忆的最大token长度
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0

#device_map = "auto"
device_map = "cpu"
#device_map = {"":"cpu"}
# if we are in a distributed setting, we need to set the device map and max memory per device
#if os.environ.get('LOCAL_RANK') is not None:
#    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
#    device_map = {'': local_rank}
def setprompt(query):
    return "GPT4 Correct User:"+query+"<|end_of_turn|>GPT4 Correct Assistant:"
from transformers import BitsAndBytesConfig 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
#    load_in_4bit=True,
    torch_dtype=torch.float16,
#    trust_remote_code=True,
#    quantization_config=BitsAndBytesConfig(
#        load_in_4bit=True,
#        bnb_4bit_compute_dtype=torch.float16,
#        bnb_4bit_use_double_quant=True,
#        bnb_4bit_quant_type="nf4",
#        llm_int8_threshold=6.0,
#        llm_int8_has_fp16_weight=False,
#    ),
)
#from peft import PeftModel, PeftConfig
#if resume_from_checkpoint:
#    model = PeftModel.from_pretrained(model, resume_from_checkpoint)
##).to(device).eval()
#the above are modified to the following for openning to training 
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
# casts all the non int8 modules to full precision (fp32) for stability
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
# 找到所有需要插入adapter的全连接层
target_modules = ['out_proj', 'down_proj', 'up_proj', 'Wqkv']
# 初始化lora配置
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
if os.path.exists(resume_from_checkpoint+'/adapter_model.bin'):
    adapters_weights = torch.load(resume_from_checkpoint+'/adapter_model.bin')
    set_peft_model_state_dict(model, adapters_weights)
model.print_trainable_parameters()
model.config.torch_dtype = torch.float32
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)
tokenizer.padding_side = "left"  # Allow batched inference
tokenizer.pad_token_id = tokenizer.bos_token_id
#print(tokenizer.pad_token_id,tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.unk_token_id,tokenizer.decode([tokenizer.pad_token_id,tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.unk_token_id]))#11320000<s><s><|end_of_turn|><unk>


# 开始对话
from transformers import StoppingCriteria,StoppingCriteriaList,MaxLengthCriteria,AdamW
import select
# define a custom stopping criteria class based on keywords
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords):
        self.keywords = keywords
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # get the last generated token
        #if input_ids[:,-1]==0:
        #    return True
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
           char = sys.stdin.read(1) 
           if char == '\x1b':#esc
               return True
        last_token = tokenizer.decode(input_ids[:, -1])
        #last_token = tokenizer.decode(input_ids[:, -1], skip_special_tokens=True) #will skip 0,which is <|endoftext|>
        # check if it is in the keywords list
        return last_token in self.keywords
stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(["<|end_of_turn|>"])])
#stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria([tokenizer.decode(tokenizer.eos_token_id)])])
#stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(['###', 'END']), MaxLengthCriteria(max_length=100)])
lianxuqa=True
#print(tokenizer.pad_token_id,tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.unk_token_id,tokenizer.decode([tokenizer.pad_token_id,tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.unk_token_id]))
# 记录所有历史记录
history_token_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
def qa(query):
    global history_token_ids 
    query=str(query)
    #print("请求：",query)
    if len(query)==0 or len(query.strip())==0:
        print("输入为空")
        return
    if lianxuqa:
        user_input_ids = tokenizer(setprompt(query), return_tensors="pt", add_special_tokens=False).input_ids
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        input_ids = history_token_ids[:, -history_max_len:].to(device)
        #input_ids = tokenizer(history + query +'<|endoftext|>' , return_tensors="pt")["input_ids"]
    else:
        input_ids = tokenizer(setprompt(query), return_tensors="pt", add_special_tokens=False).input_ids
        #input_ids = tokenizer( query+'<|endoftext|>', return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(torch.device(device))

    streamer = TextIteratorStreamer(tokenizer, timeout=180.0, skip_prompt=True)
    #streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    #gen_kwargs = generating_args.to_dict()
    gen_kwargs={}
    gen_kwargs["input_ids"] = input_ids
    gen_kwargs["max_new_tokens"]=max_new_tokens
    gen_kwargs["do_sample"]=True
    gen_kwargs["top_p"]=top_p
    gen_kwargs["temperature"]=temperature
    #gen_kwargs["repetition_penalty"]=repetition_penalty
    gen_kwargs["pad_token_id"] = 1
    #gen_kwargs["eos_token_id"] = 0
    #gen_kwargs["logits_processor"] = get_logits_processor()
    gen_kwargs["stopping_criteria"]=stopping_criteria
    gen_kwargs["streamer"] = streamer


    #with torch.no_grad():
    #    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    #    thread.start()
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    print("答：", end="", flush=True)
    response = ""
    for new_text in streamer:
        if not new_text.endswith('<|end_of_turn|>'):
            print(new_text, end="", flush=True)
            response += new_text
        else:
            print(new_text[:-15], end="", flush=True)
            response += new_text[:-15]
        #if new_text in ['<|endoftext|>']:
        #    break
    print()
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids
    eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
    response_ids = torch.concat([response_ids, eos_token_id], dim=1)
    history_token_ids = torch.concat((input_ids.cpu(), response_ids.cpu()), dim=1)
    #history = history + query +'<|endoftext|>'+ response +'<|endoftext|>'
    return response
if os.path.exists(os.environ['HOME']+'/.阿福历史对话'):
    readline.read_history_file(os.environ['HOME']+'/.阿福历史对话')
def mingling(d):
    if d[0]=="训练":
        #token=tokenizer(d[1],return_tensors="pt")
        #answer_token=tokenizer.encode(d[2],return_tensors="pt")
        #outputs=model(**token)
        #outputs.logits.shape
        #token['input_ids'].shape
        #last_token_output=outputs.logits[0,-1].view(1,-1).to(torch.device("cuda"))
        #labels=answer_token[0][0].view(1).to(torch.device("cuda"))
        #lossfct=torch.nn.CrossEntropyLoss()
        #optimizer=AdamW(model.parameters(),lr=1e-5)
        #loss=lossfct(last_token_output,labels)
        utterances=d[1:]#[q,a,q,a,q,a...]question and answer of multiturn conversation
        tmputter=[]
        for i, utterances_i in enumerate(utterances):
            if i % 2 == 0:
                tmputter.append(setprompt(utterances_i))
            else:
                tmputter.append(utterances_i)
        utterances=tmputter
        print(utterances)
        utterances_ids = tokenizer(utterances, add_special_tokens=False).input_ids
        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [tokenizer.bos_token_id]
        target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            if i % 2 == 0:
                input_ids += utterances_id 
                target_mask += [0] * (len(utterances_id) )
            else:
                input_ids += (utterances_id + [tokenizer.eos_token_id])
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        max_seq_length=2600
        input_ids = torch.tensor([input_ids[:max_seq_length]],device=device)
        target_mask = torch.tensor([target_mask[:max_seq_length]],device=device)
        attention_mask = torch.tensor([[1] * len(input_ids)],device=device)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        optimizer=AdamW(model.parameters(),lr=2e-4)
        lossfct=torch.nn.CrossEntropyLoss(ignore_index=-100)
        # 模型前馈预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        #print(input_ids,torch.round(logits[0][0]).long())
        #response = tokenizer.decode(torch.round(logits[0][0]).long())
        #print("：" + response[0].strip().replace(tokenizer.eos_token, ""))

        # 将labels中不属于target的部分，设为ignore_index:-100，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, -100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = lossfct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        model.train()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print("训练已完成")
        input_ids = tokenizer(setprompt(d[1]), return_tensors="pt", add_special_tokens=False).input_ids
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
        input_ids = torch.concat([input_ids, eos_token_id], dim=1)
        input_ids = input_ids.to(torch.device(device))
        output=model.generate(input_ids=input_ids,max_new_tokens=20)
        response = tokenizer.batch_decode(output)
        print(response[0].strip())
def 江松训练(conversation):
    # 收集多轮对话
    utterances = []
    for x in conversation:
        utterances.append(setprompt(x['instruction']) if len(x["context"])==0 else setprompt(x['instruction']+"\n"+x['context']))
        utterances.append(x['response'])
    print(utterances)
    utterances_ids = tokenizer(utterances, add_special_tokens=False).input_ids
    # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
    input_ids = [tokenizer.bos_token_id]
    target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
    for i, utterances_id in enumerate(utterances_ids):
        if i % 2 == 0:
            input_ids += utterances_id 
            target_mask += [0] * (len(utterances_id) )
        else:
            input_ids += (utterances_id + [tokenizer.eos_token_id])
            target_mask += [1] * (len(utterances_id) + 1)
    assert len(input_ids) == len(target_mask)
    # 对长度进行截断
    max_seq_length=2600
    input_ids = torch.tensor([input_ids[:max_seq_length]],device=device)
    target_mask = torch.tensor([target_mask[:max_seq_length]],device=device)
    attention_mask = torch.tensor([[1] * len(input_ids)],device=device)
    assert len(input_ids) == len(target_mask) == len(attention_mask)
    optimizer=AdamW(model.parameters(),lr=2e-4)
    lossfct=torch.nn.CrossEntropyLoss(ignore_index=-100)
    # 模型前馈预测
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    #print(input_ids,torch.round(logits[0][0]).long())
    #response = tokenizer.decode(torch.round(logits[0][0]).long())
    #print("：" + response[0].strip().replace(tokenizer.eos_token, ""))

    # 将labels中不属于target的部分，设为ignore_index:-100，只计算target部分的loss
    labels = torch.where(target_mask == 1, input_ids, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss = lossfct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    model.train()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("训练已完成")
def 训练(duihua):
    utterances=duihua#[q,a,q,a,q,a...]question and answer of multiturn conversation
    tmputter=[]
    for i, utterances_i in enumerate(utterances):
        if i % 2 == 0:
            tmputter.append(setprompt(utterances_i))
        else:
            tmputter.append(utterances_i)
    utterances=tmputter
    print(utterances)
    utterances_ids = tokenizer(utterances, add_special_tokens=False).input_ids
    # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
    input_ids = [tokenizer.bos_token_id]
    target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
    for i, utterances_id in enumerate(utterances_ids):
        if i % 2 == 0:
            input_ids += utterances_id 
            target_mask += [0] * (len(utterances_id) )
        else:
            input_ids += (utterances_id + [tokenizer.eos_token_id])
            target_mask += [1] * (len(utterances_id) + 1)
    assert len(input_ids) == len(target_mask)
    # 对长度进行截断
    max_seq_length=2600
    input_ids = torch.tensor([input_ids[:max_seq_length]],device=device)
    target_mask = torch.tensor([target_mask[:max_seq_length]],device=device)
    attention_mask = torch.tensor([[1] * len(input_ids)],device=device)
    assert len(input_ids) == len(target_mask) == len(attention_mask)
    optimizer=AdamW(model.parameters(),lr=2e-4)
    lossfct=torch.nn.CrossEntropyLoss(ignore_index=-100)
    # 模型前馈预测
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    #print(input_ids,torch.round(logits[0][0]).long())
    #response = tokenizer.decode(torch.round(logits[0][0]).long())
    #print("：" + response[0].strip().replace(tokenizer.eos_token, ""))

    # 将labels中不属于target的部分，设为ignore_index:-100，只计算target部分的loss
    labels = torch.where(target_mask == 1, input_ids, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss = lossfct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    model.train()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("训练已完成")
#    output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#    response = tokenizer.batch_decode(output)
#    print(response[0].strip())
#while True:
#    inputQ=input("问：")
#    if len(inputQ)==0:
#        print("输入为空！")
#        continue
#    if inputQ=="退出":
#        readline.write_history_file(os.environ['HOME']+'/.阿福历史对话')
#        break
#    try:
#
#        t=os.popen('python3 /media/amo/e/chat/自然派 '+inputQ).readlines()
#        if len(t)==1 and t[0][:-1]==inputQ:
#            pass
#        elif t[0][:-1]==inputQ.replace("\\",""):
#            pass
#        else:
#            for i in t:
#                print(i)
#            continue
#    except:
#        pass
#    if inputQ=="清空历史":
#        #print(history)
#        history_token_ids = torch.tensor([[]], dtype=torch.long)
#        print("已清空历史")
#        continue
#    if inputQ.startswith("【命令模式】"):
#        mingling(inputQ.split(" ")[1:])
#        continue
#    if inputQ=="连续对话":
#        lianxuqa=not lianxuqa
#        print("连续对话状态：",lianxuqa)
#        continue
#    history_token_ids=qa(inputQ,history_token_ids)
#def 问答(wen):
#    return qa(wen,torch.tensor([[]], dtype=torch.long))
#问答("你好")
#训练(["锄禾日当午","汗滴禾下土"])
def 连续对话(t):
    lianxuqa=t
    print("连续对话状态",lianxuqa)
def 清空历史(t):
    history_token_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
函子={"问答":qa,"训练":训练,"江松训练":江松训练,"连续对话":连续对话,"清空历史":清空历史}#接受一个变量的函数字典
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
