```python
from transformers import AutoModelForMaskedLM
# 加载中文bert模型
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
```


    model.safetensors:   0%|          | 0.00/412M [00:00<?, ?B/s]


    Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).



```python
# 显示模型配置信息
model.config
```




    BertConfig {
      "_name_or_path": "bert-base-chinese",
      "architectures": [
        "BertForMaskedLM"
      ],
      "attention_probs_dropout_prob": 0.1,
      "classifier_dropout": null,
      "directionality": "bidi",
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "pad_token_id": 0,
      "pooler_fc_size": 768,
      "pooler_num_attention_heads": 12,
      "pooler_num_fc_layers": 3,
      "pooler_size_per_head": 128,
      "pooler_type": "first_token_transform",
      "position_embedding_type": "absolute",
      "transformers_version": "4.38.2",
      "type_vocab_size": 2,
      "use_cache": true,
      "vocab_size": 21128
    }




```python
# 显示模型结构
model.parameters
```




<div style="max-width:800px; border: 1px solid var(--colab-border-color);"><style>
      pre.function-repr-contents {
        overflow-x: auto;
        padding: 8px 12px;
        max-height: 500px;
      }

      pre.function-repr-contents.function-repr-contents-collapsed {
        cursor: pointer;
        max-height: 100px;
      }
    </style>
    <pre style="white-space: initial; background:
         var(--colab-secondary-surface-color); padding: 8px 12px;
         border-bottom: 1px solid var(--colab-border-color);"><b>torch.nn.modules.module.Module.parameters</b><br/>def parameters(recurse: bool=True) -&gt; Iterator[Parameter]</pre><pre class="function-repr-contents function-repr-contents-collapsed" style=""><a class="filepath" style="display:none" href="#">/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py</a>Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    &gt;&gt;&gt; # xdoctest: +SKIP(&quot;undefined vars&quot;)
    &gt;&gt;&gt; for param in model.parameters():
    &gt;&gt;&gt;     print(type(param), param.size())
    &lt;class &#x27;torch.Tensor&#x27;&gt; (20L,)
    &lt;class &#x27;torch.Tensor&#x27;&gt; (20L, 1L, 5L, 5L)</pre>
      <script>
      if (google.colab.kernel.accessAllowed && google.colab.files && google.colab.files.view) {
        for (const element of document.querySelectorAll('.filepath')) {
          element.style.display = 'block'
          element.onclick = (event) => {
            event.preventDefault();
            event.stopPropagation();
            google.colab.files.view(element.textContent, 2171);
          };
        }
      }
      for (const element of document.querySelectorAll('.function-repr-contents')) {
        element.onclick = (event) => {
          event.preventDefault();
          event.stopPropagation();
          element.classList.toggle('function-repr-contents-collapsed');
        };
      }
      </script>
      </div>



## 2 加载词元化工具


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenizer
```




    BertTokenizerFast(name_or_path='bert-base-chinese', vocab_size=21128, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
    	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    }




```python
sent1 = '我爱机器学习'
sent2 = '我更爱深度学习'
#编码两个句子
encode_result = tokenizer.encode(
    text=sent1,
    text_pair=sent2,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=15,
    return_tensors=None,
)
print(encode_result)
```

    [101, 2769, 4263, 3322, 1690, 2110, 739, 102, 2769, 3291, 4263, 3918, 2428, 2110, 102]



```python
tokenizer.decode(encode_result)
```




    '[CLS] 我 爱 机 器 学 习 [SEP] 我 更 爱 深 度 学 [SEP]'




```python
#获取字典
mydict = tokenizer.get_vocab()

type(mydict), len(mydict), '强化' in mydict,
```




    (dict, 21128, False)




```python
#添加新词
tokenizer.add_tokens(new_tokens=['强化', '学习'])

#添加新符号
tokenizer.add_special_tokens({'eos_token': '[EOS]'})

mydict = tokenizer.get_vocab()

type(mydict), len(mydict), mydict['强化'], mydict['[EOS]']
```




    (dict, 21131, 21128, 21130)




```python
#编码新添加的词
encode_result = tokenizer.encode(
    text='学习强化学习[EOS]',
    text_pair=None,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=10,
    return_tensors=None,
)

print(encode_result)

tokenizer.decode(encode_result)
```

    [101, 21129, 21128, 21129, 21130, 102, 0, 0, 0, 0]





    '[CLS] 学习 强化 学习 [EOS] [SEP] [PAD] [PAD] [PAD] [PAD]'




```python

```
