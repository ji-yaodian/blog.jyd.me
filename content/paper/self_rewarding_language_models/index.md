---
title: 看论文：Self-Rewarding Language Models
date: 2024-01-30T11:45:08+08:00
tags: ['大模型', 'DPO', '强化学习', '看论文']
categories: ['看论文']
---

### 概述

语言模型通常的训练方法是先收集一大堆人类的反馈,然后基于这些反馈教模型“说话”。但这种依赖外部信号的机制缺点也很明显,模型的能力受限于人类反馈的数据指令。

所以论文提出,我们得让模型自己动手试错、自我完善。具体想法是让模型给自己当老师,让它边生成回复边给自己打分。这样模型就可以根据自己的评价,找出好和不好的回答,进而再基于这些评分来改进模型。

论文里面迭代模型的过程是这样的:

* Model0: 没有微调的预训练模型
* Model1: 基于人类反馈数据的微调模型,使用[SFT](https://arxiv.org/abs/2308.06259)的方法微调
* Model2: 使用Model1生成的回复,然后使用Model1对回复进行打分,选出好的和不好的结果,用这些结果使用[DPO](https://arxiv.org/abs/2305.18290)的方法对Model2进行微调
* Model3: 使用Model2生成的回复,然后使用Model2对回复进行打分,选出好的和不好的结果,用这些结果使用DPO的方法对Model3进行微调

这样,就可以不断的迭代下去,直到模型的能力达到预期的水平。

### 模型迭代细节

1. **Model0**:原始预训练模型

2. **Model1**:基于人类反馈数据的微调模型,使用[SFT](https://arxiv.org/abs/2308.06259)的方法微调

3. **Model2**:基于Model1自评分微调
   
    1. 生成新的指令,具体的方法参考[Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)和[Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689)
    
    2. 基于生成的指令,让Model1给每个输入生成N个回复
    
    3. 使用Model1对每个回复进行打分,返回的分数是0-5分。使用如下的Prompt:
    
        <details><summary>查看评分Prompt</summary><pre><code>Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion: 
        - Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content. 
        - Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer. 
        - Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results. 
        - Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus. 
        - Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer. 
    
        User: <INSTRUCTION_HERE> 
    
        <response><RESPONSE_HERE></response> 
    
        After examining the user’s instruction and the response: 
        - Briefly justify your total score, up to 100 words.
        - Conclude with the score using the format: “Score: <total points>” 
    
        Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necess</code></pre></details>
    
        翻译成中文：
    
        <details><summary>查看翻译成中文的Prompt</summary><pre><code>将用户的指令和相应的回复使用下面描述的5分评分系统进行对比。根据每个标准的满意度，积累得分：
    
        如果回应与用户的查询相关并提供一些与之相关的信息，即使不完整或包含一些无关内容，加1分。
        如果回应涵盖了用户问题的大部分内容，但没有完全解决问题或提供直接答案，再加1分。
        如果回应以有用的方式回答了用户问题的基本要点，无论是否看起来像是由AI助手编写，或者是否包含通常在博客或搜索结果中找到的元素，再加1分。
        如果回应明确以AI助手的角度编写，直接而全面地回答了用户问题，组织良好并且有帮助，即使在清晰度、简洁度或焦点方面仍有轻微改进的空间，再加1分。
        如果回应完美地根据用户的问题量身定制，没有多余的信息，反映出专业知识，展示了高质量、引人入胜且见地深刻的回答，再加1分。
        用户：<指令在这里>
    
        <回复><回应在这里></回复>
    
        在审查用户的指令和回应后：
    
        请简要说明您的总分，最多100字。
        以以下格式给出评分：“得分：<总分>”
        请记住，从AI助手的角度进行评估，根据需要使用网络搜索知识。</code></pre></details>
    
    4. 之后就是使用打分后的数据对模型进行训练。论文也调研了两种训练方法:
    
        * 一种是只取评分高的回复,然后使用SFT的方法进行微调
        * 另一种是取评分最高和最低的回复,构成一对训练数据,然后使用DPO的方法进行微调。论文发现,使用DPO的方法进行微调的效果更好。论文提到构造训练对的方法参考了[Some things are more cringe than others](https://arxiv.org/abs/2312.16682)。

### 后话

论文提到的方法，模型可以不断的迭代下去，直到模型的能力达到预期的水平。

创新点在于,奖励模型不是固定的，而是随着指令遵循能力的增强而变得更强。如果评分的质量越来越高,模型就可以为自己创造出超过人类反馈质量的训练数据。这是一个可能的突破口。

当然,这个方法也有一定局限。比如无限迭代训练是否会导致奖励机制的“欺骗”,模型只追求形式化的高分而不考虑真正的质量;迭代改进的上限如何等。而且，论文提到初始的种子训练数据对结果也很关键。

整体而言,语言模型通过自评分实现自我训练是一个非常有趣的方向。随着进一步探索，相信会有更多惊喜。

