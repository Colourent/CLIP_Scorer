import gradio as gr
import torch
import clip
from PIL import Image
import numpy as np
import jieba
import os
from translate import Translator

# 全局变量
_model = None
_preprocess = None
_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator(to_lang="en", from_lang="zh")
    return _translator

def translate_text(text):
    """将中文文本翻译为英文"""
    try:
        translator = get_translator()
        return translator.translate(text)
    except Exception as e:
        print(f"翻译失败: {str(e)}")
        return text

def get_model_and_processor():
    global _model, _preprocess
    if _model is None or _preprocess is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model, _preprocess = clip.load("ViT-B/32", device=device)
            _model.eval()
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise e
    return _model, _preprocess

def analyze_prompt(prompt):
    """智能分析提示词，提取关键概念"""
    words = list(jieba.cut(prompt))
    
    concepts = []
    # 添加原始完整提示词
    concepts.append({"text": prompt, "weight": 1.0, "type": "完整描述"})
    
    # 提取场景描述
    scene_words = ["傍晚", "夜晚", "黄昏", "海边", "沙滩", "黎明", "日落"]
    scenes = [w for w in words if w in scene_words]
    if scenes:
        scene_text = "".join(scenes)
        concepts.append({"text": scene_text, "weight": 0.8, "type": "场景"})
    
    # 提取人物描述
    person_words = ["老人", "年轻人", "情侣", "男人", "女人", "爷爷", "奶奶"]
    persons = [w for w in words if w in person_words]
    if persons:
        person_text = "".join(persons)
        concepts.append({"text": person_text, "weight": 0.8, "type": "人物"})
    
    # 提取动作描述
    action_words = ["手牵手", "手拉手", "散步", "走路", "漫步"]
    actions = [w for w in words if w in action_words]
    if actions:
        action_text = "".join(actions)
        concepts.append({"text": action_text, "weight": 0.6, "type": "动作"})
    
    # 翻译所有概念
    for concept in concepts:
        concept["text_en"] = translate_text(concept["text"])
    
    return concepts

def calculate_clip_scores(image, concepts, model, preprocess, device):
    """计算加权CLIP分数"""
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        scores = []
        for concept in concepts:
            # 使用英文文本进行评分
            text = concept["text_en"]
            text_tokens = clip.tokenize([text]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * (image_features @ text_features.T)).item()
            weighted_score = similarity * concept["weight"]
            
            scores.append({
                "text": concept["text"],  # 显示中文
                "text_en": text,  # 保存英文翻译
                "type": concept["type"],
                "raw_score": similarity,
                "weighted_score": weighted_score
            })
    
    return scores

def process_single_image(image, prompt):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = get_model_and_processor()
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        concepts = analyze_prompt(prompt)
        scores = calculate_clip_scores(image, concepts, model, preprocess, device)
        
        # 生成详细报告
        report = []
        
        # 计算加权总分
        total_weight = sum(concept["weight"] for concept in concepts)
        weighted_total = sum(score["weighted_score"] for score in scores)
        final_score = weighted_total / total_weight
        report.append(f"总分: {final_score:.2f}/100")
        
        # 添加完整描述的分数
        main_score = next(s for s in scores if s["type"] == "完整描述")
        report.append(f"整体匹配度: {main_score['raw_score']:.2f}")
        report.append(f"(英文翻译: {main_score['text_en']})")
        
        # 按类型组织其他分数
        score_by_type = {}
        for score in scores:
            if score["type"] != "完整描述":
                if score["type"] not in score_by_type:
                    score_by_type[score["type"]] = []
                score_by_type[score["type"]].append(score)
        
        # 添加分类报告
        for type_name, type_scores in score_by_type.items():
            report.append(f"\n{type_name}:")
            for score in type_scores:
                report.append(f"  {score['text']}: {score['raw_score']:.2f}")
                report.append(f"  (英文翻译: {score['text_en']})")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"错误: {str(e)}"

# 创建Gradio界面
iface = gr.Interface(
    fn=process_single_image,
    inputs=[
        gr.Image(type="pil", label="上传生成的图像"),
        gr.Textbox(label="输入提示词", placeholder="请输入生成图像时使用的提示词...")
    ],
    outputs=gr.Textbox(label="评估结果", lines=15),
    title="文生图评估工具",
    description="上传一张图像并输入对应的提示词，获取详细的CLIP评分分析。\n系统会智能分析场景、人物、动作等不同方面的匹配度。"
)

if __name__ == "__main__":
    iface.launch()