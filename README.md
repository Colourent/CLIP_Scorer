# CLIP_Scorer

基于CLIP模型的文生图评分工具，用于评估AI生成图像与文本提示词的匹配程度。

## 功能特点

- 支持中文提示词评估
- 智能分析场景、人物、动作等多个维度
- 自动中英文翻译
- 提供加权评分系统
- 友好的Gradio界面

## 实际效果

![评分示例](WechatIMG249.jpg)

输入提示词：傍晚海边，两位头发白花的老年手牵手

评分结果：
```
总分: 26.50/100
整体匹配度: 28.50
(英文翻译: In the evening, by the sea, two elderly people with gray hair hold hands)

场景:
  傍晚海边: 24.53
  (英文翻译: Evening by the sea)

动作:
  手牵手: 25.81
  (英文翻译: Hand in hand)
```

[其余内容保持不变...]
