---
title: "WURSTCHEN"
pubDatetime: 2024-03-15T17:31:50+08:00
tags: ["stable diffusion", "文生图"]
description: "WURSTCHEN 论文笔记：一种高效的文本到图像生成三阶段架构"
draft: true
---

论文原文
```markdown
ABSTRACT
We introduce Wurstchen, a novel architecture for text-to-image synthesis that ¨
combines competitive performance with unprecedented cost-effectiveness for largescale text-to-image diffusion models. A key contribution of our work is to develop
a latent diffusion technique in which we learn a detailed but extremely compact
semantic image representation used to guide the diffusion process. This highly
compressed representation of an image provides much more detailed guidance
compared to latent representations of language and this significantly reduces the
computational requirements to achieve state-of-the-art results. Our approach also
improves the quality of text-conditioned image generation based on our user
preference study. The training requirements of our approach consists of 24,602
A100-GPU hours – compared to Stable Diffusion 2.1's 200,000 GPU hours. Our
approach also requires less training data to achieve these results. Furthermore,
our compact latent representations allows us to perform inference over twice as
fast, slashing the usual costs and carbon footprint of a state-of-the-art (SOTA)
diffusion model significantly, without compromising the end performance. In
a broader comparison against SOTA models our approach is substantially more
efficient and compares favorably in terms of image quality. We believe that this
work motivates more emphasis on the prioritization of both performance and
computational accessibility
```

论文笔记：
## 摘要

本文研究了使用较少计算资源训练的模型与使用大量计算资源训练的模型的性能比较。结果显示，这种方法的可行性，并暗示了其可能有效地扩展到更大的模型参数。我们希望我们的工作能够为进一步研究更可持续和计算效率更高的生成式AI领域提供起点，并为在消费者硬件上训练、微调和部署大规模模型开辟更多可能性。我们将在GitHub上提供所有的源代码，包括训练和推理脚本以及训练的模型。
