# <center>Viitor-Voice</center>
### <center>An LLM based TTS engine</center>

<p align="center">
  <img src="asserts/post.webp" alt="Viitor-Voice Cover">
</p>

## 特色

- **轻量级设计**  

模型简单高效，兼容大部分LLM推理引擎; 参数规模仅为0.5B，实现了在保持高性能的同时，对计算资源的极致优化。这种设计使得模型不仅适用于服务器端，还能轻松部署在移动设备和边缘计算环境中，满足各种端上部署需求。

- **实时流式输出，低延迟体验** 

模型支持实时语音生成，能够满足对低延迟有严格要求的应用场景。在Tesla T4平台上，可实现业界领先的200ms首帧延迟，为用户提供几乎无感知的即时反馈，特别适合需要快速响应的交互式应用。

- **丰富的音色库** 

提供超过300种不同的音色选项，可以根据自己的需求和偏好，选择最合适的语音风格。无论是正式的商务演讲，还是轻松的娱乐内容，模型都能提供完美的语音匹配。

- **灵活的语速调节** 

模型支持自然语速的变化，用户可以根据内容的需要和听众的偏好，轻松调节语速。无论是加速以提高信息传递效率，还是减速以增强表达的情感深度，都能保持语音的自然流畅。

- **Zero-shot语音克隆技术（预研中）** 

Decoder only结构天然支持Zero-shot克隆, 未来将支持基于极少量语音样本的快速语音克隆.


---

## 输出样本

以下为本项目生成语音的示例：

- 示例 1: [女声-1](asserts/female_normal.wav)
- 示例 2: [男声-1](asserts/male_normal.wav)

---

## 环境准备

```commandline
conda create -n viitor_voice python=3.10
conda activate viitor_voice
pip install -r requirements.txt

### Due to the issue with vllm's tokenizer length calculation, the token limit cannot take effect.
python_package_path=`pip show pip | egrep Location | awk -F ' ' '{print $2}'`
cp viitor_voice/utils/patch.py $python_package_path/vllm/entrypoints/openai/logits_processors.py
```

---

## 推理

### 离线推理

```python
from viitor_voice.utils.offline_inference import OfflineInference
import torchaudio

tts_engine = OfflineInference(model_path='ZzWater/viitor-voice-en',
                              config_path='viitor_voice/inference_configs/en.json')
text_list = [
    "Isn't it fascinating to think about the endless possibilities that lie within the pages of a book. every time you turn a page, you're diving into a new world ripe with potential for discovery, and wonder what stories will you uncover today."]
# list valid speakers
print(tts_engine.prompt_map.keys())
audios = tts_engine.batch_infer(text_list=text_list, speaker=['1'], speed=2)
torchaudio.save('test.wav', audios[0], 24000)

```

### 流式推理(TODO)


---
## 训练(TODO)

## 参考

- [SNAC](https://github.com/hubertsiuzdak/snac)
- [mini-omni](https://github.com/gpt-omni/mini-omni)
- [open-gpt-4-o](https://laion.ai/notes/open-gpt-4-o/)

## 开源许可

本项目采用 [CC BY-NC 4.0 许可](https://creativecommons.org/licenses/by-nc/4.0/)。  
您可以自由地分享和修改本项目的代码，但仅限于非商业用途，并需遵守以下条件：

1. **署名**：您必须给予适当的署名，提供指向许可的链接，并注明是否对原作进行了修改。
2. **非商业**：不得将本项目用于商业目的。

**版权声明：**  
© 2024 Livedata. All Rights Reserved.