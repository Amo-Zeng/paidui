# 派对paidui
利用自然派来实现大语言模型对话、微调和函数调用。
# 使用方法
克隆项目后进入目录，然后运行：python dist/自然派 派对

其中自然派是用pytho编写的编程语言(详见https://github.com/Amo-Zeng/ziranpai )，派对是用自然派语言编写的交互界面，它通过调用python语言写的模块来打开大语言模型。具体参考模型模块.py。我这里用的是openchat3.5(https://huggingface.co/openchat/openchat_3.5 )，大家可以换成其他模型。 等待模型加载

完成后就会自动进入对话模式，可以直接和大模型对话。也可以输入 切换模式 来切换成命令模式，这时可以执行自然派的命令，比如：问答 你好吗？这时就会调用大语言模型来回复你好吗。训练 数组，就会把数组的奇数项当作问题，偶数项当作回答来训练模型。还有江松训练等，具体参见展示视频（https://www.bilibili.com/video/BV13x4y1e79R ）。<video src="https://www.bilibili.com/video/BV13x4y1e79R">/video>


下面展示两张截图
<center class ='img'>
<img title="派对程序运行效果" src="./Screenshot_20231220_204020992_视频.jpg" width="40%"><img title="派对程序运行效果" src="./Screenshot_20231220_204043260_视频.jpg" width="40%">
</center>



# 函数调用

我直接用prompt使大模型输出 执行自然派命令，然后只要检测到了回答开始两字是执行，我就调用自然派去执行自然派命令，然后把执行结果返回给大语言模型再次回答。这个地方用自然派命令而不是api请求有以下好处：

1.节省token,如果用api调用的话一堆网址很费token。比如要查天气的话我只要返回 搜索天气 湛江 就好了。

2.如果要对大语言模型训练输出调用指令的话，api网址更新后训练语料不需要更新了，因为网址已经隐藏在 搜索天气 湛江 里面了。

3.自然派的语法接近自然语言，更加契合大语言模型。

4. 待补充……

# 说明

我在视频中使用了bge去生成arxiv数据库，这里就没上传了。

我在github的codespace中试了一下，requirements.txt中的包有冲突，我时间有限，大家自己解决一下吧。

模型模型改自Firefly(https://github.com/yangjianxin1/Firefly/tree/master ),因为只有一张12G的显卡，所以用的cpu推理微调，改成显卡应该会更快。


